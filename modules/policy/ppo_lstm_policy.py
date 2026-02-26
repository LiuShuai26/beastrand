# modules/policy/ppo_lstm_policy.py

import numpy as np
from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from modules.model.basic_model import MLP, init_weights
from modules.model.distributions import DiagGaussianDistribution
from modules.policy.base_policy import BasePolicy


class PPOLSTMPolicy(BasePolicy):
    """Recurrent PPO policy with an LSTM core."""

    def __init__(self, cfg, activation=nn.Tanh):
        super().__init__(cfg)
        ortho_init = True

        self.body = MLP(self.obs_dim, cfg.args.mlp_layers, activation=activation)
        latent_dim = int(self.body.out_dim)

        hidden = int(getattr(cfg.args, "lstm_hidden_size", latent_dim))
        self.rnn = nn.LSTM(latent_dim, hidden)
        latent_dim = hidden

        self.dist_head = DiagGaussianDistribution(latent_dim, self.act_dim)
        self.value_head = nn.Linear(latent_dim, 1)

        self.use_lstm = True

        if ortho_init:
            module_gains = {
                self.body: np.sqrt(2),
                self.dist_head: 0.01,
                self.value_head: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(partial(init_weights, gain=gain))

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        hidden = self.rnn.hidden_size
        h = torch.zeros(1, batch_size, hidden, device=device)
        c = torch.zeros(1, batch_size, hidden, device=device)
        return h, c

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False):  # type: ignore[override]
        return self.act(inputs, deterministic=deterministic)

    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = inputs["obs"]
        latent = self.body(x)

        state = inputs.get("rnn_state")
        if state is None:
            state = self.initial_state(x.shape[0])
        h, c = state

        mask = inputs.get("mask")
        if mask is not None:
            m = mask.view(1, -1, 1)
            h = h * m
            c = c * m

        latent, (h, c) = self.rnn(latent.unsqueeze(0), (h, c))
        latent = latent.squeeze(0)

        dist = self.dist_head(latent)
        action = dist.get_actions(deterministic=deterministic)
        logp = dist.log_prob(action)
        value = self.value_head(latent).squeeze(-1)

        return {"action": action, "logp": logp, "value": value, "rnn_state": (h, c)}

    def supports_value(self) -> bool:
        return True

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs["obs"]
        latent = self.body(x)

        state = inputs.get("rnn_state")
        if state is None:
            state = self.initial_state(x.shape[0])
        h, c = state

        mask = inputs.get("mask")
        if mask is not None:
            m = mask.view(1, -1, 1)
            h = h * m
            c = c * m

        latent, _ = self.rnn(latent.unsqueeze(0), (h, c))
        latent = latent.squeeze(0)

        return self.value_head(latent).squeeze(-1)

    def evaluate_actions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs["obs"]
        action = inputs["action"]
        latent = self.body(x)

        state = inputs.get("rnn_state")
        if state is None:
            state = self.initial_state(x.shape[0])
        h, c = state

        mask = inputs.get("mask")
        if mask is not None:
            m = mask.view(1, -1, 1)
            h = h * m
            c = c * m

        latent, (h, c) = self.rnn(latent.unsqueeze(0), (h, c))
        latent = latent.squeeze(0)

        dist = self.dist_head(latent)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(latent).squeeze(-1)

        return {"logp": logp, "entropy": entropy, "value": value, "rnn_state": (h, c)}

    def build_optimizers(self, ctx, eps: float = 1e-5) -> dict:
        opt = optim.Adam(self.parameters(), lr=ctx.args.learning_rate, eps=eps)
        return {"opt": opt}
