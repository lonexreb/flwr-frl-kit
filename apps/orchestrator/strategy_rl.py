from __future__ import annotations
import io
import logging
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch as t
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

logger = logging.getLogger(__name__)


class RLFedAvgStrategy(fl.server.strategy.FedAvg):
    """Federated Averaging strategy adapted for RL model aggregation.

    This strategy aggregates neural network weights from RL clients,
    handling PyTorch model serialization and providing RL-specific
    configuration options.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        # RL-specific parameters
        steps_per_round: int = 1000,
        eval_episodes: int = 5,
        staleness_threshold: int = 3,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.steps_per_round = steps_per_round
        self.eval_episodes = eval_episodes
        self.staleness_threshold = staleness_threshold
        self.client_staleness: Dict[str, int] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {
            "steps": self.steps_per_round,
            "server_round": server_round,
        }
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {
            "episodes": self.eval_episodes,
            "server_round": server_round,
        }
        if self.on_evaluate_config_fn is not None:
            config.update(self.on_evaluate_config_fn(server_round))

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using federated averaging."""
        if not results:
            return None, {}

        # Log failures
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")

        # Filter out clients that might be too stale
        filtered_results = []
        for client, fit_res in results:
            client_id = client.cid
            if client_id in self.client_staleness:
                self.client_staleness[client_id] += 1
            else:
                self.client_staleness[client_id] = 0

            # Skip clients that are too stale (haven't participated recently)
            if self.client_staleness[client_id] <= self.staleness_threshold:
                filtered_results.append((client, fit_res))
                self.client_staleness[client_id] = 0  # Reset staleness
            else:
                logger.warning(f"Skipping stale client {client_id}")

        if not filtered_results:
            logger.warning("No non-stale clients available for aggregation")
            return None, {}

        # Extract weights and number of examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in filtered_results
        ]

        # Perform federated averaging
        aggregated_ndarrays = aggregate_weights(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in filtered_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        else:
            # Default aggregation: weighted average of training metrics
            metrics_aggregated = self._aggregate_training_metrics(filtered_results)

        logger.info(f"Round {server_round}: Aggregated {len(filtered_results)} client updates")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Log failures
        if failures:
            logger.warning(f"Round {server_round} eval: {len(failures)} client failures")

        # Default aggregation: weighted average of evaluation metrics
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        else:
            metrics_aggregated = self._aggregate_evaluation_metrics(results)

        # Use average return as the loss (negated for minimization-oriented APIs)
        avg_return = metrics_aggregated.get("avg_return", 0.0)
        loss_aggregated = -float(avg_return)  # Negate because Flower expects loss

        logger.info(f"Round {server_round} eval: Average return = {avg_return:.2f}")

        return loss_aggregated, metrics_aggregated

    def _aggregate_training_metrics(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[str, Scalar]:
        """Aggregate training metrics using weighted averages."""
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        if total_examples == 0:
            return {}

        aggregated = {}

        # Collect all metric keys
        all_keys = set()
        for _, fit_res in results:
            all_keys.update(fit_res.metrics.keys())

        # Aggregate each metric
        for key in all_keys:
            if key == "steps":  # Sum steps across clients
                aggregated[key] = sum(
                    fit_res.metrics.get(key, 0) for _, fit_res in results
                )
            else:  # Weighted average for other metrics
                weighted_sum = sum(
                    fit_res.metrics.get(key, 0) * fit_res.num_examples
                    for _, fit_res in results
                )
                aggregated[key] = weighted_sum / total_examples

        return aggregated

    def _aggregate_evaluation_metrics(
        self, results: List[Tuple[ClientProxy, EvaluateRes]]
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics using weighted averages."""
        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        if total_examples == 0:
            return {}

        aggregated = {}

        # Collect all metric keys
        all_keys = set()
        for _, eval_res in results:
            all_keys.update(eval_res.metrics.keys())

        # Aggregate each metric
        for key in all_keys:
            if key == "episodes":  # Sum episodes across clients
                aggregated[key] = sum(
                    eval_res.metrics.get(key, 0) for _, eval_res in results
                )
            else:  # Weighted average for other metrics
                weighted_sum = sum(
                    eval_res.metrics.get(key, 0) * eval_res.num_examples
                    for _, eval_res in results
                )
                aggregated[key] = weighted_sum / total_examples

        return aggregated


def aggregate_weights(weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Aggregate model weights using federated averaging.

    Args:
        weights_results: List of tuples containing (weights, num_examples)

    Returns:
        List of aggregated weight arrays
    """
    if not weights_results:
        return []

    # Calculate total number of examples
    total_examples = sum(num_examples for _, num_examples in weights_results)

    if total_examples == 0:
        logger.warning("Total examples is 0, using simple averaging")
        total_examples = len(weights_results)
        weights_results = [(weights, 1) for weights, _ in weights_results]

    # Initialize aggregated weights with zeros
    first_weights, _ = weights_results[0]
    aggregated_weights = [np.zeros_like(w) for w in first_weights]

    # Aggregate weights using weighted average
    for weights, num_examples in weights_results:
        weight = num_examples / total_examples
        for i, layer_weights in enumerate(weights):
            aggregated_weights[i] += layer_weights * weight

    return aggregated_weights