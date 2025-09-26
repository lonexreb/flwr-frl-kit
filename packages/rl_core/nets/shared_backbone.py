from __future__ import annotations
import torch as t
import torch.nn as nn

class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)
