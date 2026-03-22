# projects/ppo_lstm/policy.py

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from beastrand.core.model.basic_model import MLP
from beastrand.core.model.distributions import DiagGaussianDistribution
from beastrand.core.base_policy import BasePolicy


class PPOLSTMPolicy(BasePolicy):
    """Recurrent PPO policy with an LSTM core."""

    def __init__(self, cfg, activation=nn.Tanh):
        super().__init__(cfg)

        self.body = MLP(self.obs_dim, cfg.args.mlp_layers, activation=activation)
        latent_dim = int(self.body.out_dim)

        hidden = int(getattr(cfg.args, "lstm_hidden_size", latent_dim))
        self.rnn = nn.LSTM(latent_dim, hidden)
        latent_dim = hidden

        self.dist_head = DiagGaussianDistribution(latent_dim, self.act_dim)
        self.value_head = nn.Linear(latent_dim, 1)

        self.use_lstm = True

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        hidden = self.rnn.hidden_size
        h = torch.zeros(1, batch_size, hidden, device=device)
        c = torch.zeros(1, batch_size, hidden, device=device)
        return h, c

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False):  # type: ignore[override]
        return self.act(inputs, deterministic=deterministic)

    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        x = self.normalize_obs(inputs["obs"])
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

        out = {"action": action, "logp": logp, "value": value, "rnn_state": (h, c)}
        if hasattr(dist, "action_logits"):
            out["action_logits"] = dist.action_logits()
        return out

    def supports_value(self) -> bool:
        return True

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.normalize_obs(inputs["obs"])
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
        x = self.normalize_obs(inputs["obs"])
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

        out = {"logp": logp, "entropy": entropy, "value": value, "rnn_state": (h, c)}
        if hasattr(dist, "action_logits"):
            out["action_logits"] = dist.action_logits()
        return out

    def evaluate_sequences(
        self,
        obs_flat: torch.Tensor,
        actions_flat: torch.Tensor,
        rnn_states_flat: torch.Tensor,
        dones_flat: torch.Tensor,
        recurrence: int,
    ) -> Dict[str, torch.Tensor]:
        """PackedSequence BPTT: split at episode boundaries, single LSTM call.

        Args:
            obs_flat: (N, obs_dim) — flat observations for all S*T steps
            actions_flat: (N, act_dim) — flat actions
            rnn_states_flat: (N, hidden*2) — per-step h+c concatenated
            dones_flat: (N,) — done flags (1.0 = episode ended)
            recurrence: int — rollout length T
        Returns:
            dict with logp, entropy, value — all (N,) flat
        """
        from projects.ppo_lstm.rnn_utils import build_rnn_inputs, unpack_rnn_outputs

        # 1. Batched encoder: normalize + MLP body for all N samples at once
        x = self.normalize_obs(obs_flat)
        body_out = self.body(x)

        # 2. PackedSequence LSTM — splits at episode boundaries automatically
        input_seq, rnn_h0, inv_inds = build_rnn_inputs(
            body_out, dones_flat, rnn_states_flat, recurrence,
        )

        # Split concatenated h+c into separate LSTM state tensors
        hidden = self.rnn.hidden_size
        h0, c0 = rnn_h0[:, :hidden], rnn_h0[:, hidden:]
        h0 = h0.unsqueeze(0).contiguous()
        c0 = c0.unsqueeze(0).contiguous()

        output_seq, _ = self.rnn(input_seq, (h0.detach(), c0.detach()))
        core_out = unpack_rnn_outputs(output_seq, inv_inds)

        # 3. Batched decoder: dist + value heads for all N outputs at once
        dist = self.dist_head(core_out)
        logp = dist.log_prob(actions_flat)
        entropy = dist.entropy()
        value = self.value_head(core_out).squeeze(-1)

        result = {"logp": logp, "entropy": entropy, "value": value}
        if hasattr(dist, "action_logits"):
            result["action_logits"] = dist.action_logits()
        return result

    def build_optimizers(self, ctx, eps: float = 1e-6) -> dict:
        opt = optim.Adam(self.parameters(), lr=ctx.args.learning_rate, eps=eps)
        return {"opt": opt}
