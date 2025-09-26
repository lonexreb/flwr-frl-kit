#!/usr/bin/env python3
"""Flower client for federated RL training using A2C."""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import flwr as fl

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.rl_core.client_runtime.a2c_client import A2CClient, A2CConfig
from packages.rl_core.client_runtime.flower_adapter import FlowerClientAdapter

import os
os.environ["ANSI_COLORS_DISABLED"] = "1"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Flower RL client."""
    parser = argparse.ArgumentParser(description="Flower Federated RL Client")

    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost:8080",
        help="Server address (default: localhost:8080)",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client ID (default: 0)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Environment ID (default: CartPole-v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None, uses client_id + 42)",
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
        help="Evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--dummy-weight-update",
        action="store_true",
        help="Use dummy weight updates for testing server aggregation (default: False)",
    )

    args = parser.parse_args()

    # Set seed based on client ID if not provided
    seed = args.seed if args.seed is not None else args.client_id + 42

    logger.info(f"Starting client {args.client_id} with env {args.env_id}, seed {seed}")
    logger.info(f"Dummy weight update mode: {args.dummy_weight_update}")

    # Create RL client configuration
    config = A2CConfig(
        env_id=args.env_id,
        seed=seed,
    )

    # Create RL client (only needed if not using dummy mode)
    if args.dummy_weight_update:
        logger.info("Dummy mode enabled - skipping RL client creation")
        rl_client = None
    else:
        logger.info("Creating A2C client")
        rl_client = A2CClient(config)

    # Wrap with Flower adapter
    flower_client = FlowerClientAdapter(
        rl_client=rl_client,
        round_train_steps=args.steps_per_round,
        eval_episodes=args.eval_episodes,
        dummy_weight_update=args.dummy_weight_update,
    )

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=flower_client,
    )


if __name__ == "__main__":
    main()