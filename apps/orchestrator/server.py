#!/usr/bin/env python3
"""Flower server for federated reinforcement learning.

This server coordinates federated learning across multiple RL clients,
aggregating their model weights and managing training rounds.
"""

from __future__ import annotations
import argparse
import logging
import sys
import io
from pathlib import Path
from typing import Dict, Optional

import flwr as fl
import torch as t
import numpy as np
from flwr.common import Context, Parameters, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.history import History

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.orchestrator.strategy_rl import RLFedAvgStrategy
from packages.rl_core.algos.a2c import A2C
from packages.rl_core.envs.make_env import make_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("flower_server.log"),
    ],
)
logger = logging.getLogger(__name__)


def fit_config(server_round: int) -> Dict[str, any]:
    """Return training configuration dict for each round.

    Args:
        server_round: The current server round number

    Returns:
        Configuration dictionary sent to clients
    """
    config = {
        "server_round": server_round,
        "steps": 1000,  # Training steps per round
    }

    # Optionally adjust training parameters based on round
    if server_round > 10:
        config["steps"] = 1500  # Increase training in later rounds

    logger.info(f"Round {server_round} fit config: {config}")
    return config


def evaluate_config(server_round: int) -> Dict[str, any]:
    """Return evaluation configuration dict for each round.

    Args:
        server_round: The current server round number

    Returns:
        Configuration dictionary sent to clients for evaluation
    """
    config = {
        "server_round": server_round,
        "episodes": 5,  # Evaluation episodes per round
    }

    logger.info(f"Round {server_round} evaluate config: {config}")
    return config


def fit_metrics_aggregation_fn(metrics_list):
    """Aggregate training metrics from multiple clients.

    Args:
        metrics_list: List of (num_examples, metrics) tuples

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics_list)

    if total_examples == 0:
        return {}

    aggregated = {}

    # Collect all metric keys
    all_keys = set()
    for _, metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        if key == "steps":  # Sum total steps across clients
            aggregated[key] = sum(
                metrics.get(key, 0) for _, metrics in metrics_list
            )
        else:  # Weighted average for other metrics
            weighted_sum = sum(
                metrics.get(key, 0) * num_examples
                for num_examples, metrics in metrics_list
            )
            aggregated[key] = weighted_sum / total_examples

    logger.info(f"Aggregated training metrics: {aggregated}")
    return aggregated


def evaluate_metrics_aggregation_fn(metrics_list):
    """Aggregate evaluation metrics from multiple clients.

    Args:
        metrics_list: List of (num_examples, metrics) tuples

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics_list)

    if total_examples == 0:
        return {}

    aggregated = {}

    # Collect all metric keys
    all_keys = set()
    for _, metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        if key == "episodes":  # Sum total episodes across clients
            aggregated[key] = sum(
                metrics.get(key, 0) for _, metrics in metrics_list
            )
        else:  # Weighted average for other metrics
            weighted_sum = sum(
                metrics.get(key, 0) * num_examples
                for num_examples, metrics in metrics_list
            )
            aggregated[key] = weighted_sum / total_examples

    logger.info(f"Aggregated evaluation metrics: {aggregated}")
    return aggregated


def create_initial_model(env_id: str = "CartPole-v1", hidden_sizes=(128, 128)) -> A2C:
    """Create and initialize the global model.

    Args:
        env_id: Environment identifier to determine observation/action space
        hidden_sizes: Hidden layer sizes for the neural network

    Returns:
        Initialized A2C model
    """
    # Create temporary environment to get dimensions
    env = make_env(env_id, seed=42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # Create model
    model = A2C(obs_dim, act_dim, hidden_sizes)

    logger.info(f"Created model for {env_id}: obs_dim={obs_dim}, act_dim={act_dim}")
    return model


def model_to_parameters(model: A2C) -> Parameters:
    """Convert PyTorch model to Flower Parameters.

    Args:
        model: PyTorch model to convert

    Returns:
        Flower Parameters object
    """
    # Serialize model state dict
    buf = io.BytesIO()
    t.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()

    # Convert to numpy array and then to Parameters
    model_array = np.frombuffer(model_bytes, dtype=np.uint8)
    return ndarrays_to_parameters([model_array])


def get_env_config(env_id: str) -> Dict[str, any]:
    """Get environment configuration to send to clients.

    Args:
        env_id: Environment identifier

    Returns:
        Environment configuration dictionary
    """
    return {
        "env_id": env_id,
        "seed": None,  # Let clients choose their own seeds for diversity
    }


def create_strategy(
    min_clients: int = 2,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    steps_per_round: int = 1000,
    eval_episodes: int = 5,
    initial_parameters: Optional[Parameters] = None,
    env_id: str = "CartPole-v1",
) -> RLFedAvgStrategy:
    """Create the federated learning strategy.

    Args:
        min_clients: Minimum number of clients required
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        steps_per_round: Training steps per round
        eval_episodes: Evaluation episodes per round

    Returns:
        Configured strategy instance
    """
    # Create fit config function that includes environment info
    def fit_config_with_env(server_round: int) -> Dict[str, any]:
        config = fit_config(server_round)
        config.update(get_env_config(env_id))
        return config

    # Create evaluate config function that includes environment info
    def evaluate_config_with_env(server_round: int) -> Dict[str, any]:
        config = evaluate_config(server_round)
        config.update(get_env_config(env_id))
        return config

    strategy = RLFedAvgStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        on_fit_config_fn=fit_config_with_env,
        on_evaluate_config_fn=evaluate_config_with_env,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
        steps_per_round=steps_per_round,
        eval_episodes=eval_episodes,
        accept_failures=True,  # Allow some client failures
    )

    logger.info(f"Created strategy with min_clients={min_clients}")
    return strategy


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    min_clients: int = 2,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    steps_per_round: int = 1000,
    eval_episodes: int = 5,
    env_id: str = "CartPole-v1",
    hidden_sizes: tuple = (128, 128),
) -> History:
    """Start the Flower server.

    Args:
        server_address: Server address and port
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients required
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        steps_per_round: Training steps per round
        eval_episodes: Evaluation episodes per round

    Returns:
        Training history
    """
    logger.info(f"Starting Flower server on {server_address}")
    logger.info(f"Configuration: rounds={num_rounds}, min_clients={min_clients}")
    logger.info(f"Environment: {env_id}, Model: {hidden_sizes}")

    # 1. Load/Create the initial model
    logger.info("Creating initial global model...")
    global_model = create_initial_model(env_id, hidden_sizes)
    initial_parameters = model_to_parameters(global_model)
    logger.info("Global model initialized")

    # 2. Create strategy without initial parameters for testing
    strategy = create_strategy(
        min_clients=min_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        steps_per_round=steps_per_round,
        eval_episodes=eval_episodes,
        initial_parameters=None,  # Disable initial parameters for now
        env_id=env_id,
    )

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)

    # Start server
    history = fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    logger.info("Server finished. Training history:")
    if history.losses_distributed:
        for round_num, (loss, _) in enumerate(history.losses_distributed, 1):
            logger.info(f"Round {round_num}: Loss = {loss:.4f}")

    return history


def main():
    """Main entry point for the Flower server."""
    parser = argparse.ArgumentParser(description="Flower Federated RL Server")

    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of clients required (default: 2)",
    )
    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=1.0,
        help="Fraction of clients to use for training (default: 1.0)",
    )
    parser.add_argument(
        "--fraction-evaluate",
        type=float,
        default=1.0,
        help="Fraction of clients to use for evaluation (default: 1.0)",
    )
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=1000,
        help="Training steps per round (default: 1000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Evaluation episodes per round (default: 5)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Environment ID for RL training (default: CartPole-v1)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="128,128",
        help="Hidden layer sizes as comma-separated values (default: 128,128)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Parse hidden sizes
    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(','))

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        history = start_server(
            server_address=args.server_address,
            num_rounds=args.rounds,
            min_clients=args.min_clients,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            steps_per_round=args.steps_per_round,
            eval_episodes=args.eval_episodes,
            env_id=args.env_id,
            hidden_sizes=hidden_sizes,
        )

        logger.info("Federated learning completed successfully!")

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()