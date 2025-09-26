from __future__ import annotations
import torch as t, torch.nn as nn, torch.optim as optim
import numpy as np
from ..nets.shared_backbone import MLPBackbone
from ..nets.heads import PolicyHead, ValueHead

class A2C(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128)):
        super().__init__()
        self.backbone = MLPBackbone(obs_dim, hidden_sizes)
        feat_dim = hidden_sizes[-1]
        self.policy = PolicyHead(feat_dim, act_dim)
        self.value = ValueHead(feat_dim)

    def forward(self, obs: t.Tensor):
        feats = self.backbone(obs)
        dist = self.policy(feats)
        val = self.value(feats)
        return dist, val

class A2CTrainer:
    def __init__(self, model: A2C, lr=3e-4, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def step(self, obs, actions, returns, advantages, old_logprobs):
        dist, values = self.model(obs)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * logprobs).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()

        with t.no_grad():
            approx_kl = (old_logprobs - logprobs).mean().clamp_min(0.0).item()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl": float(approx_kl),
        }
