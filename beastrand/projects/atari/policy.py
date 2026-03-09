# projects/atari/policy.py
"""CNN-based PPO policy for Atari environments (SF-matched)."""

from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from beastrand.core.model.basic_model import NatureCNN, init_weights
from beastrand.core.model.distributions import CategoricalDistribution
from beastrand.core.base_policy import BasePolicy


class AtariPolicy(BasePolicy):
    """NatureCNN + PPO heads for discrete Atari environments.

    SF-matched:
      - Orthogonal weight init (gain=1.0)
      - /255 scaling in _preprocess (before obs normalizer)
      - adam_eps=1e-5
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        in_channels = cfg.obs_shape[0] if len(cfg.obs_shape) == 3 else 1
        self.encoder = NatureCNN(in_channels=in_channels)
        latent_dim = self.encoder.out_dim

        self.dist_head = CategoricalDistribution(latent_dim, cfg.act_n)
        self.value_head = nn.Linear(latent_dim, 1)

        self.use_lstm = False

        # SF-style orthogonal init for all Conv2d + Linear layers
        self.apply(partial(init_weights, gain=1.0))

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """Scale uint8 images to [0, 1], then apply obs normalizer if enabled."""
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        elif obs.dtype != torch.float32:
            obs = obs.float()
        # Ensure channels-first: (B, C, H, W)
        # FrameStack produces (B, C, H, W) where C=4 (stacked frames)
        # and H=W=84, so shape[1]=4 < shape[2]=84 — already channels-first.
        return self.normalize_obs(obs.reshape(obs.shape[0], -1)).reshape(obs.shape)

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False):
        return self.act(inputs, deterministic=deterministic)

    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = self._preprocess(inputs["obs"])
        latent = self.encoder(x)

        dist = self.dist_head(latent)
        action = dist.get_actions(deterministic=deterministic)
        logp = dist.log_prob(action)
        action = action.unsqueeze(-1).float()  # [B] -> [B, 1]
        value = self.value_head(latent).squeeze(-1)

        return {"action": action, "logp": logp, "value": value}

    def supports_value(self) -> bool:
        return True

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._preprocess(inputs["obs"])
        latent = self.encoder(x)
        return self.value_head(latent).squeeze(-1)

    def evaluate_actions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._preprocess(inputs["obs"])
        action = inputs["action"]
        latent = self.encoder(x)

        dist = self.dist_head(latent)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(latent).squeeze(-1)

        return {"logp": logp, "entropy": entropy, "value": value}

    def build_optimizers(self, ctx, eps: float = 1e-5) -> dict:
        opt = optim.Adam(self.parameters(), lr=ctx.args.learning_rate, eps=eps)
        return {"opt": opt}
