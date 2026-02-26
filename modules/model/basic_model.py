import numpy as np
import torch
import torch.nn as nn


def init_weights(module: nn.Module, gain: float = 1) -> None:
    """
    Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, activation=nn.Tanh):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), activation()]
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
