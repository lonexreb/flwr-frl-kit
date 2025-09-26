from __future__ import annotations
import numpy as np
import gymnasium as gym

class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, eps: float = 1e-8):
        super().__init__(env)
        self.eps = eps
        self.running_mean = None
        self.running_var = None
        self.count = 0

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if self.running_mean is None:
            self.running_mean = np.zeros_like(obs)
            self.running_var = np.ones_like(obs)
        self.count += 1
        momentum = 1.0 / self.count
        self.running_mean = (1 - momentum) * self.running_mean + momentum * obs
        self.running_var = (1 - momentum) * self.running_var + momentum * (obs - self.running_mean) ** 2
        return (obs - self.running_mean) / (np.sqrt(self.running_var) + self.eps)
