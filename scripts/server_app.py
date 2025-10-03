"""RL Federated Learning: A Flower / TorchRL server app."""

import logging
import torch
import torch.nn as nn
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from packages.rl_core.utils import FederatedRLEnvLoader
from packages.rl_core.nets.torchrl_nets import TorchRLNetBuilder
from packages.rl_core.algos import evaluate_policy

# Create ServerApp
app = ServerApp()

# Setup logger
logger = logging.getLogger(__name__)


class TorchRLPolicy(nn.Module):
    """Policy network using TorchRL MLP backbone."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        # Use TorchRL MLP for feature extraction
        self.backbone = TorchRLNetBuilder(
            network_type="MLP",
            in_features=obs_dim,
            out_features=hidden_sizes[-1],
            num_cells=hidden_sizes[:-1],
            activation_class=nn.ReLU,
        ).get_network()

        # Policy head: outputs action logits
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.policy_head(features)
        return torch.distributions.Categorical(logits=logits)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    verbose: bool = context.run_config.get("verbose", False)

    # Parse network hidden sizes from string (comma-separated)
    hidden_sizes_str: str = context.run_config.get("network-hidden-sizes", "128,128")
    hidden_sizes: list[int] = [int(x.strip()) for x in hidden_sizes_str.split(",")]

    # Configure logging based on verbose flag
    if verbose:
        logging.basicConfig(level=logging.INFO, format='[SERVER] %(message)s')
        logger.info(f"Starting federated RL training with {num_rounds} rounds")
        logger.info(f"Configuration: lr={lr}, fraction_evaluate={fraction_evaluate}, hidden_sizes={hidden_sizes}")

    # Create test environment to get specs
    test_env = FederatedRLEnvLoader.create_partitioned_env(
        env_type="gym",
        env_name="CartPole-v1",
        partition_id=0,
        num_partitions=1,
        partition_strategy="seed",
        base_seed=42,
    )

    obs_spec = test_env.observation_spec["observation"]
    action_spec = test_env.action_spec
    obs_dim = obs_spec.shape[-1]
    action_dim = action_spec.space.n
    test_env.close()

    if verbose:
        logger.info(f"Environment specs: obs_dim={obs_dim}, action_dim={action_dim}")

    # Initialize global policy using TorchRL MLP
    global_policy = TorchRLPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes)
    arrays = ArrayRecord(global_policy.state_dict())

    if verbose:
        num_params = sum(p.numel() for p in global_policy.parameters())
        initial_global_norm = sum(p.norm().item() for p in global_policy.parameters())
        logger.info(f"Initialized global policy with {num_params:,} parameters, norm={initial_global_norm:.4f}")

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    if verbose:
        logger.info("Starting federated averaging...")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda round, arrays: global_evaluate(round, arrays, hidden_sizes, verbose),
    )

    # Save final model to disk
    if verbose:
        final_policy = TorchRLPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes)
        final_policy.load_state_dict(result.arrays.to_torch_state_dict())
        final_global_norm = sum(p.norm().item() for p in final_policy.parameters())
        global_weight_change = abs(final_global_norm - initial_global_norm)
        logger.info(f"Training complete! Global weight change: {global_weight_change:.4f}")
        logger.info("Saving final policy to disk...")
    else:
        print("\nSaving final policy to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_policy.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord, hidden_sizes: list[int], verbose: bool = False) -> MetricRecord:
    """Evaluate global policy on centralized environment (for testing purposes only).

    Warning:
        This centralized evaluation is for monitoring training progress only.
        Real federated RL should use distributed evaluation across clients.
    """

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create evaluation environment
    eval_env = FederatedRLEnvLoader.create_partitioned_env(
        env_type="gym",
        env_name="CartPole-v1",
        partition_id=0,
        num_partitions=1,
        partition_strategy="seed",
        base_seed=999,  # Different seed for unbiased evaluation
    )

    # Get environment specs
    obs_spec = eval_env.observation_spec["observation"]
    action_spec = eval_env.action_spec
    obs_dim = obs_spec.shape[-1]
    action_dim = action_spec.space.n

    # Initialize and load policy using TorchRL MLP
    policy_net = TorchRLPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes)
    policy_net.load_state_dict(arrays.to_torch_state_dict())
    policy_net.to(device)

    if verbose:
        round_weight_norm = sum(p.norm().item() for p in policy_net.parameters())
        logger.info(f"Round {server_round}: Evaluating global policy, weight_norm={round_weight_norm:.4f}")

    # Evaluate the global policy
    eval_metrics = evaluate_policy(
        policy_net=policy_net,
        env=eval_env,
        num_episodes=20,
        device=str(device),
    )

    # Clean up
    eval_env.close()

    if verbose:
        logger.info(
            f"Round {server_round} evaluation: "
            f"reward={eval_metrics['avg_episode_reward']:.2f} ± {eval_metrics['std_episode_reward']:.2f}, "
            f"length={eval_metrics['avg_episode_length']:.1f}"
        )

    # Return the evaluation metrics
    return MetricRecord({
        "avg_episode_reward": eval_metrics["avg_episode_reward"],
        "avg_episode_length": eval_metrics["avg_episode_length"],
        "std_episode_reward": eval_metrics["std_episode_reward"],
    })
