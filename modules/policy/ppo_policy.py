# modules/policy/ppo_policy.py

import numpy as np
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from modules.model.basic_model import MLP, init_weights
from modules.model.distributions import DiagGaussianDistribution
from modules.policy.base_policy import BasePolicy


class PPOPolicy(BasePolicy):
    """Feedforward PPO policy implementing the unified Policy API."""

    def __init__(self, cfg, activation=nn.Tanh):
        super().__init__(cfg)
        ortho_init = True

        self.body = MLP(self.obs_dim, cfg.args.mlp_layers, activation=activation)
        latent_dim = int(self.body.out_dim)

        self.dist_head = DiagGaussianDistribution(latent_dim, self.act_dim)
        self.value_head = nn.Linear(latent_dim, 1)

        self.use_lstm = False

        if ortho_init:
            module_gains = {
                self.body: np.sqrt(2),
                self.dist_head: 0.01,
                self.value_head: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(partial(init_weights, gain=gain))

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False):  # type: ignore[override]
        return self.act(inputs, deterministic=deterministic)

    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = inputs["obs"]
        latent = self.body(x)

        dist = self.dist_head(latent)
        action = dist.get_actions(deterministic=deterministic)
        logp = dist.log_prob(action)
        value = self.value_head(latent).squeeze(-1)

        return {"action": action, "logp": logp, "value": value}

    def supports_value(self) -> bool:
        return True

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs["obs"]
        latent = self.body(x)
        return self.value_head(latent).squeeze(-1)

    def evaluate_actions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs["obs"]
        action = inputs["action"]
        latent = self.body(x)

        dist = self.dist_head(latent)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(latent).squeeze(-1)

        return {"logp": logp, "entropy": entropy, "value": value}

    def build_optimizers(self, ctx, eps: float = 1e-5) -> dict:
        opt = optim.Adam(self.parameters(), lr=ctx.args.learning_rate, eps=eps)
        return {"opt": opt}
