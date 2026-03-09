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


class NatureCNN(nn.Module):
    """Nature DQN-style CNN encoder for image observations.

    Expects input shape [B, C, H, W] (channels-first).
    Default architecture: 3 conv layers → flatten → Linear → ReLU.
    Output: [B, out_dim] (default 512).
    """
    def __init__(self, in_channels: int = 4, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute conv output size: 84x84 -> 20x20 -> 9x9 -> 7x7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


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
