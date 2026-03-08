# ppo/policy.py

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from core.model.basic_model import MLP
from core.model.distributions import DiagGaussianDistribution, CategoricalDistribution
from core.base_policy import BasePolicy


class PPOPolicy(BasePolicy):
    """Feedforward PPO policy implementing the unified Policy API."""

    def __init__(self, cfg, activation=nn.Tanh):
        super().__init__(cfg)

        self.body = MLP(self.obs_dim, cfg.args.mlp_layers, activation=activation)
        latent_dim = int(self.body.out_dim)

        self._discrete = cfg.act_kind == "discrete"
        if self._discrete:
            self.dist_head = CategoricalDistribution(latent_dim, cfg.act_n)
        else:
            self.dist_head = DiagGaussianDistribution(latent_dim, self.act_dim)
        self.value_head = nn.Linear(latent_dim, 1)

        self.use_lstm = False

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False):  # type: ignore[override]
        return self.act(inputs, deterministic=deterministic)

    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = self.normalize_obs(inputs["obs"])
        latent = self.body(x)

        dist = self.dist_head(latent)
        action = dist.get_actions(deterministic=deterministic)
        logp = dist.log_prob(action)
        if self._discrete:
            action = action.unsqueeze(-1).float()  # [B] -> [B, 1] for shared tensor
        value = self.value_head(latent).squeeze(-1)

        out = {"action": action, "logp": logp, "value": value}
        if hasattr(dist, "action_logits"):
            out["action_logits"] = dist.action_logits()
        return out

    def supports_value(self) -> bool:
        return True

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.normalize_obs(inputs["obs"])
        latent = self.body(x)
        return self.value_head(latent).squeeze(-1)

    def evaluate_actions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.normalize_obs(inputs["obs"])
        action = inputs["action"]
        latent = self.body(x)

        dist = self.dist_head(latent)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(latent).squeeze(-1)

        out = {"logp": logp, "entropy": entropy, "value": value}
        if hasattr(dist, "action_logits"):
            out["action_logits"] = dist.action_logits()
        return out

    def build_optimizers(self, ctx, eps: float = 1e-6) -> dict:
        opt = optim.Adam(self.parameters(), lr=ctx.args.learning_rate, eps=eps)
        return {"opt": opt}
