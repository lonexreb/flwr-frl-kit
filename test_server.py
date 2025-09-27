#!/usr/bin/env python3
"""Test server script that simulates federated learning with dummy clients using Flower.

This script creates fake clients with random weights and simulates federated
learning by aggregating their weights using the existing Flower strategy.
"""

from __future__ import annotations
import logging
import random
import sys
import io
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch as t
import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, Context,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig
from flwr.simulation import start_simulation

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from apps.orchestrator.strategy_rl import RLFedAvgStrategy
from packages.rl_core.algos.a2c import A2C
from packages.rl_core.envs.make_env import make_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DummyRLClient(NumPyClient):
    """Dummy RL client that simulates training with random weight updates."""

    def __init__(self, client_id: str, env_id: str = "CartPole-v1", seed: Optional[int] = None):
        self.client_id = client_id
        self.env_id = env_id
        self.rng = np.random.default_rng(seed)

        # Create the model
        self.model = self._create_model()

        # Add some noise to simulate different client initialization
        with t.no_grad():
            for param in self.model.parameters():
                noise = t.randn_like(param) * 0.1
                param.add_(noise)

        logger.info(f"Initialized dummy client {client_id}")

    def _create_model(self) -> A2C:
        """Create the A2C model for this client."""
        env = make_env(self.env_id, seed=42)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        env.close()
        return A2C(obs_dim, act_dim, (128, 128))

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = t.from_numpy(new_param).data

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Simulate training by updating weights with random noise."""
        logger.info(f"Client {self.client_id}: Starting training round {config.get('server_round', '?')}")

        # Set the received parameters
        self.set_parameters(parameters)

        # Get training configuration
        steps = int(config.get("steps", 1000))

        # Simulate training by adding random updates to weights
        with t.no_grad():
            for param in self.model.parameters():
                # Simulate gradient-based updates with noise
                update = t.randn_like(param) * 0.01 * (steps / 1000)
                param.add_(update)

        # Generate realistic training metrics
        base_loss = self.rng.uniform(10, 30)
        noise_factor = self.rng.uniform(0.8, 1.2)

        metrics = {
            "loss": float(base_loss * noise_factor),
            "policy_loss": float(self.rng.uniform(-0.02, 0.02)),
            "value_loss": float(base_loss * 2 * noise_factor),
            "entropy": float(self.rng.uniform(0.68, 0.70)),
            "kl": float(self.rng.uniform(0.0, 0.01)),
            "steps": float(steps)
        }

        logger.info(f"Client {self.client_id}: Completed {steps} training steps")

        return self.get_parameters(config), steps, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Simulate evaluation by generating random performance metrics."""
        logger.info(f"Client {self.client_id}: Starting evaluation round {config.get('server_round', '?')}")

        # Set the received parameters
        self.set_parameters(parameters)

        # Get evaluation configuration
        episodes = int(config.get("episodes", 5))

        # Generate evaluation metrics (simulated episode returns)
        base_return = self.rng.uniform(8, 60)  # CartPole range
        eval_metrics = {
            "avg_return": float(base_return),
            "std_return": float(self.rng.uniform(5, 15)),
            "episodes": float(episodes)
        }

        # Use negative average return as loss (Flower expects loss to be minimized)
        loss = -base_return

        logger.info(f"Client {self.client_id}: Completed evaluation with avg_return={base_return:.2f}")

        return loss, episodes, eval_metrics


def dummy_client_fn(context: Context) -> Client:
    """Create a dummy client function for Flower simulation."""
    # Extract client ID from context
    cid = str(context.node_id)
    seed = hash(cid) % 10000  # Deterministic seed based on client ID
    return DummyRLClient(cid, seed=seed).to_client()


def create_initial_parameters(env_id: str = "CartPole-v1") -> Parameters:
    """Create initial parameters for the global model."""
    # Create a temporary model to get initial parameters
    env = make_env(env_id, seed=42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    model = A2C(obs_dim, act_dim, (128, 128))

    # Convert to parameters
    ndarrays = [param.detach().cpu().numpy() for param in model.parameters()]
    return ndarrays_to_parameters(ndarrays)


def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
    """Generate fit configuration for each round."""
    config = {
        "server_round": server_round,
        "steps": 1000,  # Training steps per round
        "env_id": "CartPole-v1",
    }

    # Increase steps in later rounds
    if server_round > 5:
        config["steps"] = 1500

    return config


def evaluate_config_fn(server_round: int) -> Dict[str, Scalar]:
    """Generate evaluation configuration for each round."""
    return {
        "server_round": server_round,
        "episodes": 5,  # Evaluation episodes per round
        "env_id": "CartPole-v1",
    }


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate training metrics from multiple clients."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    # Initialize aggregated metrics
    aggregated = {}

    # Collect all metric keys
    all_keys = set()
    for _, client_metrics in metrics:
        all_keys.update(client_metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        if key == "steps":  # Sum total steps
            aggregated[key] = sum(
                client_metrics.get(key, 0) for _, client_metrics in metrics
            )
        else:  # Weighted average for other metrics
            weighted_sum = sum(
                client_metrics.get(key, 0) * num_examples
                for num_examples, client_metrics in metrics
            )
            aggregated[key] = weighted_sum / total_examples

    logger.info(f"Aggregated training metrics: {aggregated}")
    return aggregated


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics from multiple clients."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    # Initialize aggregated metrics
    aggregated = {}

    # Collect all metric keys
    all_keys = set()
    for _, client_metrics in metrics:
        all_keys.update(client_metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        if key == "episodes":  # Sum total episodes
            aggregated[key] = sum(
                client_metrics.get(key, 0) for _, client_metrics in metrics
            )
        else:  # Weighted average for other metrics
            weighted_sum = sum(
                client_metrics.get(key, 0) * num_examples
                for num_examples, client_metrics in metrics
            )
            aggregated[key] = weighted_sum / total_examples

    logger.info(f"Aggregated evaluation metrics: {aggregated}")
    return aggregated


def create_strategy(num_clients: int = 4) -> RLFedAvgStrategy:
    """Create the federated learning strategy."""
    min_clients = max(2, num_clients // 2)

    strategy = RLFedAvgStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=create_initial_parameters(),
        steps_per_round=1000,
        eval_episodes=5,
        accept_failures=True,
    )

    logger.info(f"Created strategy with {num_clients} clients (min_clients={min_clients})")
    return strategy


def run_federated_simulation(
    num_clients: int = 4,
    num_rounds: int = 10,
    env_id: str = "CartPole-v1"
) -> None:
    """Run federated learning simulation using Flower."""
    logger.info(f"Starting Flower federated simulation:")
    logger.info(f"  Clients: {num_clients}")
    logger.info(f"  Rounds: {num_rounds}")
    logger.info(f"  Environment: {env_id}")

    # Create client functions
    client_resources = {"num_cpus": 1}

    # Create strategy
    strategy = create_strategy(num_clients)

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)

    # Run simulation
    history = start_simulation(
        client_fn=dummy_client_fn,
        num_clients=num_clients,
        config=config,
        strategy=strategy,
        client_resources=client_resources,
    )

    # Print results
    logger.info("\n=== Simulation Results ===")

    if history.losses_distributed:
        logger.info("Training losses per round:")
        for round_num, (loss, _) in enumerate(history.losses_distributed, 1):
            logger.info(f"  Round {round_num}: Loss = {loss:.4f}")

    if history.losses_centralized:
        logger.info("Evaluation losses per round:")
        for round_num, (loss, _) in enumerate(history.losses_centralized, 1):
            logger.info(f"  Round {round_num}: Eval Loss = {loss:.4f}")

    # Print metrics if available
    if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
        logger.info("Final training metrics:")
        try:
            final_metrics = history.metrics_distributed[-1][1]
            for key, value in final_metrics.items():
                logger.info(f"  {key}: {value}")
        except (IndexError, KeyError):
            logger.info("  No final training metrics available")

    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        logger.info("Final evaluation metrics:")
        try:
            final_metrics = history.metrics_centralized[-1][1]
            for key, value in final_metrics.items():
                logger.info(f"  {key}: {value}")
        except (IndexError, KeyError):
            logger.info("  No final evaluation metrics available")

    logger.info("Federated learning simulation completed!")
    return history


def main():
    """Main entry point for the test server."""
    logger.info("Starting Flower-based federated RL test server")

    # Run the simulation
    history = run_federated_simulation(
        num_clients=4,
        num_rounds=10,
        env_id="CartPole-v1"
    )

    logger.info("Test server simulation completed successfully!")


if __name__ == "__main__":
    main()