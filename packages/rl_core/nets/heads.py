from __future__ import annotations
import torch as t
import torch.nn as nn

class PolicyHead(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.logits = nn.Linear(in_dim, n_actions)

    def forward(self, feats: t.Tensor) -> t.distributions.Categorical:
        return t.distributions.Categorical(logits=self.logits(feats))

class ValueHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.v = nn.Linear(in_dim, 1)

    def forward(self, feats: t.Tensor) -> t.Tensor:
        return self.v(feats).squeeze(-1)
