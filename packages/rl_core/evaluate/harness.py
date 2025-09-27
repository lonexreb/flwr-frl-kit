#!/usr/bin/env python3
"""Evaluation harness for saved RL models."""

from __future__ import annotations
import argparse
import torch as t
import numpy as np
from pathlib import Path
import sys

# Add packages to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from packages.rl_core.algos.a2c import A2C
from packages.rl_core.envs.make_env import make_env


def evaluate_checkpoint(
    ckpt_path: str,
    env_id: str = "CartPole-v1",
    episodes: int = 5,
    seed: int = 42,
    render: bool = False
) -> dict:
    """Evaluate a saved checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        env_id: Environment ID
        episodes: Number of evaluation episodes
        seed: Random seed
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load environment to get dimensions
    env = make_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Create and load model
    model = A2C(obs_dim, act_dim, hidden_sizes=(128, 128))
    checkpoint = t.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Evaluate
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        
        while not done:
            with t.no_grad():
                obs_t = t.as_tensor(obs, dtype=t.float32).unsqueeze(0)
                dist, _ = model(obs_t)
                action = dist.probs.argmax(dim=-1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            
            if render:
                env.render()
        
        returns.append(ep_return)
        print(f"Episode {ep + 1}: Return = {ep_return:.1f}")
    
    env.close()
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"\nAverage Return: {avg_return:.2f} ± {std_return:.2f}")
    
    return {
        "avg_return": float(avg_return),
        "std_return": float(std_return),
        "episodes": episodes,
        "returns": returns
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--env_id", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    args = parser.parse_args()
    
    if not Path(args.ckpt).exists():
        print(f"Error: Checkpoint not found: {args.ckpt}")
        sys.exit(1)
    
    evaluate_checkpoint(
        ckpt_path=args.ckpt,
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render
    )


if __name__ == "__main__":
    main()
