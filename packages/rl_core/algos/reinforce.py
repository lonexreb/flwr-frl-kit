from __future__ import annotations
import torch
import torch.nn as nn
from torchrl.envs import EnvBase
from typing import Callable


def train_reinforce(
    policy_net: nn.Module,
    env: EnvBase,
    num_episodes: int,
    lr: float,
    gamma: float = 0.99,
    device: str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Train a policy network using REINFORCE algorithm.

    This is a simple, efficient implementation of the REINFORCE (policy gradient)
    algorithm that works with any PyTorch policy network that returns a
    torch.distributions.Distribution when called.

    Algorithm:
        1. Collect episode trajectory using current policy
        2. Compute discounted returns (Monte Carlo)
        3. Normalize returns for stability
        4. Compute policy gradient: ∇J = E[∇log π(a|s) * G]
        5. Update policy parameters

    Args:
        policy_net: Policy network that outputs action distribution.
            Forward pass should return torch.distributions.Distribution
        env: TorchRL environment
        num_episodes: Number of episodes to train
        lr: Learning rate
        gamma: Discount factor (default: 0.99)
        device: Device to train on (default: "cpu")
        optimizer: Optional pre-configured optimizer. If None, uses Adam.

    Returns:
        Dictionary containing training metrics:
            - avg_episode_reward: Average reward across episodes
            - avg_episode_length: Average episode length
            - total_episodes: Number of episodes trained

    Example:
        >>> env = FederatedRLEnvLoader.create_partitioned_env(
        ...     env_type="gym", env_name="CartPole-v1", partition_id=0, num_partitions=1
        ... )
        >>> policy = TorchRLPolicy(obs_dim=4, action_dim=2)
        >>> metrics = train_reinforce(policy, env, num_episodes=100, lr=0.001)
        >>> print(f"Avg reward: {metrics['avg_episode_reward']}")
    """
    policy_net.to(device)

    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        # Reset environment
        td = env.reset()

        log_probs = []
        rewards = []
        done = False
        step_count = 0

        # Collect episode trajectory
        while not done and step_count < 1000:
            # Get state
            state = td["observation"].to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Get action distribution from policy
            action_dist = policy_net(state)

            # Sample action and get log probability
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)

            # Take action in environment
            td["action"] = action.squeeze().cpu()
            td = env.step(td)

            # Store reward (reward is in the 'next' key after step)
            reward = td["next"]["reward"].item()
            rewards.append(reward)

            # Check if episode is done (check terminated or done in 'next' after step)
            done = td["next"].get("done", td["next"].get("terminated", torch.tensor(False))).item()
            step_count += 1

        # Calculate discounted returns (Monte Carlo)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=device)

        # Normalize returns for stability (reduces variance)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate policy gradient loss
        # REINFORCE gradient: ∇J = E[∇log π(a|s) * G]
        # We minimize negative expected return
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        # Update policy
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()

        # Check gradient magnitudes (for first episode only to avoid spam)
        if episode == 0:
            total_grad_norm = sum(p.grad.norm().item() for p in policy_net.parameters() if p.grad is not None)
            print(f"Episode 0: loss={loss.item():.4f}, grad_norm={total_grad_norm:.4f}, returns_mean={returns.mean().item():.4f}")

        optimizer.step()

        # Track metrics
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

    return {
        "avg_episode_reward": sum(episode_rewards) / len(episode_rewards),
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths),
        "total_episodes": num_episodes,
        "total_steps": sum(episode_lengths),
    }


def evaluate_policy(
    policy_net: nn.Module,
    env: EnvBase,
    num_episodes: int = 10,
    device: str = "cpu",
    deterministic: bool = False,
) -> dict[str, float]:
    """Evaluate a policy network.

    Args:
        policy_net: Policy network to evaluate
        env: TorchRL environment
        num_episodes: Number of episodes to evaluate (default: 10)
        device: Device to evaluate on (default: "cpu")
        deterministic: If True, use mode() instead of sample() for actions

    Returns:
        Dictionary containing evaluation metrics:
            - avg_episode_reward: Average reward across episodes
            - avg_episode_length: Average episode length
            - std_episode_reward: Standard deviation of rewards
            - num_episodes: Number of episodes evaluated

    Example:
        >>> metrics = evaluate_policy(policy, env, num_episodes=20)
        >>> print(f"Avg reward: {metrics['avg_episode_reward']:.2f} ± {metrics['std_episode_reward']:.2f}")
    """
    policy_net.to(device)
    policy_net.eval()

    episode_rewards = []
    episode_lengths = []

    with torch.no_grad():
        for episode in range(num_episodes):
            # Reset environment
            td = env.reset()

            rewards = []
            done = False
            step_count = 0

            while not done and step_count < 1000:
                # Get state
                state = td["observation"].to(device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)

                # Get action from policy
                action_dist = policy_net(state)

                # Use deterministic action or sample
                if deterministic and hasattr(action_dist, 'mode'):
                    action = action_dist.mode
                else:
                    action = action_dist.sample()

                # Take action
                td["action"] = action.squeeze().cpu()
                td = env.step(td)

                # Track reward (reward is in the 'next' key after step)
                reward = td["next"]["reward"].item()
                rewards.append(reward)

                # Check if done (check terminated or done in 'next' after step)
                done = td["next"].get("done", td["next"].get("terminated", torch.tensor(False))).item()
                step_count += 1

            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)

    policy_net.train()

    return {
        "avg_episode_reward": sum(episode_rewards) / len(episode_rewards),
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths),
        "std_episode_reward": torch.tensor(episode_rewards).std().item() if len(episode_rewards) > 1 else 0.0,
        "num_episodes": num_episodes,
        "total_steps": sum(episode_lengths),
    }
