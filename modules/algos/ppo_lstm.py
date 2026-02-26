"""PPO learner with truncated backprop for LSTM policies."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .ppo import compute_gae, normalize_advantages, to_torch


class PPOLSTMAlgorithm:
    def __init__(self, ctx, policy, opt, device):
        self.ctx = ctx
        self.policy = policy
        self.opt = opt
        self.device = device

    def prepare_batch(self, slot_view):
        return compute_gae(self.ctx, slot_view)

    def update(self, batch):
        return ppo_lstm_update(self.ctx, self.policy, self.opt, batch, self.device)

    def save_checkpoint(self, save_dir: str, policy: nn.Module) -> None:
        """Save policy weights (ONNX export not supported for LSTM policies)."""
        os.makedirs(save_dir, exist_ok=True)

        policy_path = os.path.join(save_dir, "policy.pt")
        torch.save(policy.state_dict(), policy_path)
        logging.info("saved policy to %s", policy_path)
        logging.info("ONNX export skipped for LSTM policy (not supported)")


def _validate_recurrence(ctx, N: int) -> int:
    recurrence = int(getattr(ctx.args, "recurrence", -1))
    if recurrence <= 0:
        recurrence = int(getattr(ctx.args, "rollout", getattr(ctx.args, "rollout_t", 1)))
    recurrence = max(1, min(recurrence, max(1, N)))
    if ctx.args.minibatch_size % recurrence != 0:
        raise ValueError("minibatch_size must be divisible by recurrence for LSTM training")
    return recurrence


def _reshape_sequences(tensor: torch.Tensor, total_sequences: int, recurrence: int) -> torch.Tensor:
    return tensor.reshape(total_sequences, recurrence, *tensor.shape[1:])


def ppo_lstm_update(
    ctx,
    policy: nn.Module,
    opt: Dict[str, optim.Optimizer],
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    policy.train()

    optimizer = opt["opt"]
    data = to_torch(batch, device)

    if not bool(getattr(policy, "use_lstm", False)):
        raise ValueError("PPOLSTMAlgorithm requires a recurrent policy (use_lstm=True)")

    b_obs = data["obs"]
    b_actions = data["act"]
    b_logprobs = data["logp"].float()
    b_advantages = data["adv"].float()
    b_returns = data["ret"].float()
    b_values = data["val"].float()
    b_rnn_h = data.get("rnn_state_h")
    b_rnn_c = data.get("rnn_state_c")
    b_mask = data.get("mask")

    if b_rnn_h is None or b_rnn_c is None:
        raise ValueError("LSTM PPO requires rnn_state_h and rnn_state_c tensors")

    N = b_obs.shape[0]
    if N == 0:
        raise ValueError("ppo_lstm_update received an empty batch")

    recurrence = _validate_recurrence(ctx, N)

    total_sequences = N // recurrence
    if total_sequences <= 0:
        raise ValueError("Not enough samples to form a recurrent minibatch")

    seq_obs = _reshape_sequences(b_obs, total_sequences, recurrence)
    seq_actions = _reshape_sequences(b_actions, total_sequences, recurrence)
    seq_logprobs = _reshape_sequences(b_logprobs, total_sequences, recurrence)
    seq_advantages = _reshape_sequences(b_advantages, total_sequences, recurrence)
    seq_returns = _reshape_sequences(b_returns, total_sequences, recurrence)
    seq_values = _reshape_sequences(b_values, total_sequences, recurrence)
    seq_rnn_h = _reshape_sequences(b_rnn_h, total_sequences, recurrence)
    seq_rnn_c = _reshape_sequences(b_rnn_c, total_sequences, recurrence)
    seq_mask = _reshape_sequences(b_mask, total_sequences, recurrence) if b_mask is not None else None

    seq_batch_size = ctx.args.minibatch_size // recurrence
    if seq_batch_size == 0:
        raise ValueError("minibatch_size must be at least as large as recurrence for LSTM training")

    clipfracs = []
    approx_kl = torch.tensor(0.0, device=device)
    old_approx_kl = torch.tensor(0.0, device=device)

    seq_inds = np.arange(total_sequences)
    n_mb = total_sequences / seq_batch_size
    for epoch in range(ctx.args.train_epochs):
        np.random.shuffle(seq_inds)
        for start in range(0, total_sequences, seq_batch_size):
            end = start + seq_batch_size
            mb_seq = seq_inds[start:end]
            if len(mb_seq) == 0:
                continue

            obs_mb = seq_obs[mb_seq]
            act_mb = seq_actions[mb_seq]
            logprob_old_mb = seq_logprobs[mb_seq]
            adv_mb = seq_advantages[mb_seq]
            ret_mb = seq_returns[mb_seq]
            val_old_mb = seq_values[mb_seq]
            h0 = seq_rnn_h[mb_seq, 0]
            c0 = seq_rnn_c[mb_seq, 0]
            mask_mb = seq_mask[mb_seq] if seq_mask is not None else None

            state = (h0.unsqueeze(0).detach(), c0.unsqueeze(0).detach())
            h, c = state
            logp_news = []
            entropies = []
            values = []

            for t in range(recurrence):
                step_inputs: Dict[str, Any] = {
                    "obs": obs_mb[:, t, ...],
                    "action": act_mb[:, t, ...],
                    "rnn_state": (h, c),
                }
                if mask_mb is not None:
                    step_inputs["mask"] = mask_mb[:, t]

                out = policy.evaluate_actions(step_inputs)
                logp_news.append(out["logp"])
                entropies.append(out["entropy"])
                values.append(out["value"])
                h, c = out.get("rnn_state", (h, c))

            newlogprob = torch.stack(logp_news, dim=1)
            entropy = torch.stack(entropies, dim=1)
            newvalue = torch.stack(values, dim=1)

            logratio = newlogprob - logprob_old_mb
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > ctx.args.ppo_clip_range).float().mean().item())

            mb_advantages = adv_mb
            if ctx.args.normalize_adv:
                mb_advantages = normalize_advantages(mb_advantages)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - ctx.args.ppo_clip_range, 1 + ctx.args.ppo_clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue_flat = newvalue.reshape(-1)
            ret_flat = ret_mb.reshape(-1)
            val_old_flat = val_old_mb.reshape(-1)
            v_loss_unclipped = (newvalue_flat - ret_flat) ** 2
            v_clipped = val_old_flat + torch.clamp(
                newvalue_flat - val_old_flat,
                -ctx.args.ppo_clip_value,
                ctx.args.ppo_clip_value,
            )
            v_loss_clipped = (v_clipped - ret_flat) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ctx.args.entropy_coef * entropy_loss + v_loss * ctx.args.value_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), ctx.args.max_grad_norm)
            optimizer.step()

    return {
        "pi_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "adv_mean": b_advantages.mean().item(),
        "adv_std": b_advantages.std(unbiased=False).item(),
        "value_mean": b_values.mean().item(),
        "value_std": b_values.std(unbiased=False).item(),
        "entropy_coef": float(ctx.args.entropy_coef),
        "approx_kl": approx_kl.item(),
        "clip_frac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "num_minibatches": float(n_mb),
    }
