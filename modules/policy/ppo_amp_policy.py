# modules/policy/ppo_amp_policy.py
"""
PPO-AMP policy with dual value heads (task + style).

Same actor as PPOPolicy but with two independent critics — one for the
environment task reward and one for the discriminator style reward.
Returns ``value`` as a ``[B, 2]`` tensor so that multi-group GAE and
value loss can be computed per reward group.
"""

import numpy as np
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from modules.model.basic_model import MLP, init_weights
from modules.model.distributions import DiagGaussianDistribution
from modules.policy.base_policy import BasePolicy


G = 2  # reward groups: 0 = task, 1 = style


class PPOAMPPolicy(BasePolicy):
    """PPO-AMP actor-critic with dual value heads."""

    def __init__(self, cfg, activation=nn.Tanh):
        super().__init__(cfg)
        ortho_init = True

        self.body = MLP(self.obs_dim, cfg.args.mlp_layers, activation=activation)
        latent_dim = int(self.body.out_dim)

        self.dist_head = DiagGaussianDistribution(latent_dim, self.act_dim)

        # Dual critics: task (group 0) and style (group 1)
        self.value_heads = nn.ModuleList([
            nn.Linear(latent_dim, 1) for _ in range(G)
        ])

        self.use_lstm = False

        if ortho_init:
            module_gains = {
                self.body: np.sqrt(2),
                self.dist_head: 0.01,
            }
            for module, gain in module_gains.items():
                module.apply(partial(init_weights, gain=gain))
            for vh in self.value_heads:
                vh.apply(partial(init_weights, gain=1.0))

    def _get_values(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute per-group values from latent features.

        Returns:
            Tensor of shape (B, 2).
        """
        return torch.cat([vh(latent) for vh in self.value_heads], dim=-1)

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False):
        return self.act(inputs, deterministic=deterministic)

    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = inputs["obs"]
        latent = self.body(x)

        dist = self.dist_head(latent)
        action = dist.get_actions(deterministic=deterministic)
        logp = dist.log_prob(action)
        value = self._get_values(latent)  # (B, 2)

        return {"action": action, "logp": logp, "value": value}

    def supports_value(self) -> bool:
        return True

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs["obs"]
        latent = self.body(x)
        return self._get_values(latent)  # (B, 2)

    def evaluate_actions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs["obs"]
        action = inputs["action"]
        latent = self.body(x)

        dist = self.dist_head(latent)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self._get_values(latent)  # (B, 2)

        return {"logp": logp, "entropy": entropy, "value": value}

    def build_optimizers(self, ctx, eps: float = 1e-5) -> dict:
        opt = optim.Adam(self.parameters(), lr=ctx.args.learning_rate, eps=eps)
        return {"opt": opt}
