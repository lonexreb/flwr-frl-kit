"""RL Federated Learning: A Flower / TorchRL app."""

import logging
import torch
import torch.nn as nn
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from packages.rl_core.utils import FederatedRLEnvLoader
from packages.rl_core.nets.torchrl_nets import TorchRLNetBuilder
from packages.rl_core.algos import train_reinforce, evaluate_policy

# Flower ClientApp
app = ClientApp()

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


@app.train()
def train(msg: Message, context: Context):
    """Train the RL policy on local environment using REINFORCE."""

    # Get config from run config
    verbose: bool = context.run_config.get("verbose", False)
    partition_id = context.node_config["partition-id"]

    # Parse network hidden sizes from string (comma-separated)
    hidden_sizes_str: str = context.run_config.get("network-hidden-sizes", "128,128")
    hidden_sizes: list[int] = [int(x.strip()) for x in hidden_sizes_str.split(",")]

    if verbose:
        logging.basicConfig(level=logging.INFO, format=f'[CLIENT {partition_id}] %(message)s')
        logger.info(f"Starting training with hidden_sizes={hidden_sizes}...")

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load RL environment
    num_partitions = context.node_config["num-partitions"]

    env = FederatedRLEnvLoader.create_partitioned_env(
        env_type="gym",
        env_name="CartPole-v1",
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition_strategy="seed",
        base_seed=42,
    )

    if verbose:
        logger.info(f"Created environment with seed {42 + partition_id}")

    # Get environment specs
    obs_spec = env.observation_spec["observation"]
    action_spec = env.action_spec

    obs_dim = obs_spec.shape[-1]
    action_dim = action_spec.space.n

    # Initialize policy network using TorchRL MLP
    policy_net = TorchRLPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes)

    # Track initial weights
    initial_weight_norm = sum(p.norm().item() for p in policy_net.parameters())

    # Load weights if provided
    if "arrays" in msg.content:
        policy_net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        initial_weight_norm = sum(p.norm().item() for p in policy_net.parameters())
        if verbose:
            logger.info(f"Loaded global model weights, norm={initial_weight_norm:.4f}")
    else:
        if verbose:
            logger.warning("⚠️ No arrays in message - using random initialization!")

    # Training configuration
    num_episodes = context.run_config.get("local-epochs", 10)
    lr = msg.content.get("config", {}).get("lr", 0.001)

    if verbose:
        logger.info(f"Training for {num_episodes} episodes with lr={lr}")

    # Train using REINFORCE algorithm
    train_metrics = train_reinforce(
        policy_net=policy_net,
        env=env,
        num_episodes=num_episodes,
        lr=lr,
        gamma=0.99,
        device=str(device),
    )

    if verbose:
        final_weight_norm = sum(p.norm().item() for p in policy_net.parameters())
        weight_change = abs(final_weight_norm - initial_weight_norm) if "arrays" in msg.content else final_weight_norm
        logger.info(
            f"Training complete: "
            f"reward={train_metrics['avg_episode_reward']:.2f}, "
            f"length={train_metrics['avg_episode_length']:.1f}, "
            f"steps={train_metrics['total_steps']}, "
            f"weight_norm={final_weight_norm:.4f}, "
            f"change={weight_change:.4f}"
        )

    # Construct and return reply Message
    model_record = ArrayRecord(policy_net.state_dict())
    metrics = {
        "avg_episode_reward": train_metrics["avg_episode_reward"],
        "avg_episode_length": train_metrics["avg_episode_length"],
        "num_episodes": train_metrics["total_episodes"],
        "num-examples": train_metrics["total_steps"],  # Required by Flower for weighted averaging
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    env.close()
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the RL policy on local environment."""

    # Get config from run config
    verbose: bool = context.run_config.get("verbose", False)
    partition_id = context.node_config["partition-id"]

    # Parse network hidden sizes from string (comma-separated)
    hidden_sizes_str: str = context.run_config.get("network-hidden-sizes", "128,128")
    hidden_sizes: list[int] = [int(x.strip()) for x in hidden_sizes_str.split(",")]

    if verbose:
        logging.basicConfig(level=logging.INFO, format=f'[CLIENT {partition_id}] %(message)s')
        logger.info("Starting evaluation...")

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load RL environment
    num_partitions = context.node_config["num-partitions"]

    env = FederatedRLEnvLoader.create_partitioned_env(
        env_type="gym",
        env_name="CartPole-v1",
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition_strategy="seed",
        base_seed=42,
    )

    # Get environment specs
    obs_spec = env.observation_spec["observation"]
    action_spec = env.action_spec

    obs_dim = obs_spec.shape[-1]
    action_dim = action_spec.space.n

    # Initialize and load policy network using TorchRL MLP
    policy_net = TorchRLPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes)
    policy_net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Evaluate policy
    eval_metrics = evaluate_policy(
        policy_net=policy_net,
        env=env,
        num_episodes=10,
        device=str(device),
    )

    if verbose:
        logger.info(
            f"Evaluation complete: "
            f"reward={eval_metrics['avg_episode_reward']:.2f} ± {eval_metrics['std_episode_reward']:.2f}, "
            f"length={eval_metrics['avg_episode_length']:.1f}"
        )

    # Construct and return reply Message
    metrics = {
        "avg_episode_reward": eval_metrics["avg_episode_reward"],
        "avg_episode_length": eval_metrics["avg_episode_length"],
        "std_episode_reward": eval_metrics["std_episode_reward"],
        "num_episodes": eval_metrics["num_episodes"],
        "num-examples": eval_metrics["total_steps"],  # Required by Flower for weighted averaging
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    env.close()
    return Message(content=content, reply_to=msg)
