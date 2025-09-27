# packages/rl_core/client_runtime/a2c_client.py
from __future__ import annotations
import io, random, logging
import numpy as np
import torch as t
from dataclasses import dataclass
from typing import Dict
from ..algos.a2c import A2C, A2CTrainer
from ..algos.utils import compute_gae
from ..envs.make_env import make_env

@dataclass
class A2CConfig:
    env_id: str = "CartPole-v1"
    seed: int = 17
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: tuple = (128,128)

class A2CClient:
    def __init__(self, cfg: A2CConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        random.seed(cfg.seed)
        t.manual_seed(cfg.seed)
        self.env = make_env(cfg.env_id, seed=cfg.seed)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.model = A2C(obs_dim, act_dim, cfg.hidden_sizes)
        self.trainer = A2CTrainer(
            self.model, lr=cfg.lr, vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef, max_grad_norm=cfg.max_grad_norm
        )

        self.last_obs, _ = self.env.reset(seed=cfg.seed)
        self.last_done = False
        self.total_steps = 0
        self.logger = logging.getLogger(f"A2CClient-{cfg.env_id}")
        self.logger.setLevel(logging.INFO)

    # --- serialization helpers ---
    def _state_dict_to_bytes(self, state_dict) -> bytes:
        buf = io.BytesIO()
        t.save(state_dict, buf)
        return buf.getvalue()

    def _bytes_to_state_dict(self, blob: bytes):
        buf = io.BytesIO(blob)
        return t.load(buf, weights_only=False, map_location="cpu")

    # --- RLClient API ---
    def get_weights(self) -> Dict[str, bytes]:
        sd = self.model.state_dict()
        return {"model": self._state_dict_to_bytes(sd)}

    def set_weights(self, w: Dict[str, bytes]) -> None:
        if "model" in w:
            self.model.load_state_dict(self._bytes_to_state_dict(w["model"]))

    def _rollout(self, T: int):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        for _ in range(T):
            # Reset environment if already done from previous rollout
            if self.last_done:
                self.last_obs, _ = self.env.reset()
                self.last_done = False

            obs_t = t.as_tensor(self.last_obs, dtype=t.float32).unsqueeze(0)
            dist, value = self.model(obs_t)
            action = dist.sample().item()
            logp = dist.log_prob(t.tensor(action)).item()

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            obs_buf.append(self.last_obs)
            act_buf.append(action)
            rew_buf.append(reward)
            done_buf.append(float(done))
            val_buf.append(value.squeeze(0).item())
            logp_buf.append(logp)

            self.last_obs = next_obs
            self.last_done = done
            if done:
                self.last_obs, _ = self.env.reset()
                self.last_done = False

        # bootstrap value
        with t.no_grad():
            next_obs_t = t.as_tensor(self.last_obs, dtype=t.float32).unsqueeze(0)
            _, next_v = self.model(next_obs_t)
            next_v = next_v.item()
        return (
            t.tensor(np.array(obs_buf), dtype=t.float32),
            t.tensor(np.array(act_buf), dtype=t.long),
            t.tensor(np.array(rew_buf), dtype=t.float32),
            t.tensor(np.array(done_buf), dtype=t.float32),
            t.tensor(np.array(val_buf), dtype=t.float32),
            t.tensor(np.array(logp_buf), dtype=t.float32),
            t.tensor(next_v, dtype=t.float32),
        )

    def train_for(self, steps: int) -> dict:
        # collect at least `steps` transitions using chunks of rollout_len
        total = 0
        logs = {}
        while total < steps:
            (obs, acts, rews, dones, vals, old_logp, next_v) = self._rollout(self.cfg.rollout_len)
            adv, ret = compute_gae(rews, dones, vals, next_v, self.cfg.gamma, self.cfg.gae_lambda)
            # normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            logs = self.trainer.step(obs, acts, ret, adv, old_logp)
            total += len(rews)

            # Update total steps and log every 1000 steps
            prev_milestone = self.total_steps // 1000
            self.total_steps += len(rews)
            current_milestone = self.total_steps // 1000

            if current_milestone > prev_milestone:
                self.logger.info(f"Step {self.total_steps}: loss={logs.get('loss', 'N/A'):.4f}, "
                               f"policy_loss={logs.get('policy_loss', 'N/A'):.4f}, "
                               f"value_loss={logs.get('value_loss', 'N/A'):.4f}, "
                               f"entropy={logs.get('entropy', 'N/A'):.4f}")

        return {
            "steps": int(total),
            **logs
        }

    def evaluate(self, episodes: int = 5) -> dict:
        returns = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                with t.no_grad():
                    dist, _ = self.model(t.as_tensor(obs, dtype=t.float32).unsqueeze(0))
                    action = dist.probs.argmax(dim=-1).item()
                obs, r, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_ret += float(r)
            returns.append(ep_ret)
        return {
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "episodes": episodes,
        }
