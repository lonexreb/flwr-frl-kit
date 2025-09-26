from __future__ import annotations
import gymnasium as gym

def make_env(env_id: str, seed: int | None = None):
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
    return env
