"""PPO learner with truncated backprop for LSTM policies."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from beastrand.ppo.algorithm import compute_gae, normalize_advantages, to_torch


class PPOLSTMAlgorithm:
    def __init__(self, ctx, policy, opt, device):
        self.ctx = ctx
        self.policy = policy
        self.opt = opt
        self.device = device

        self.returns_rms = None
        if getattr(ctx.args, "normalize_returns", False):
            from beastrand.core.running_mean_std import RunningMeanStd
            self.returns_rms = RunningMeanStd()

        # LR schedule
        self._initial_lr = ctx.args.learning_rate
        self._lr_schedule = getattr(ctx.args, "lr_schedule", "constant")
        self._lr_update_count = 0
        self._lr_total_updates = max(1, ctx.args.total_env_steps // ctx.args.batch_size)

    def _step_lr(self):
        """Apply learning rate schedule before each update."""
        if self._lr_schedule == "linear_decay":
            frac = 1.0 - self._lr_update_count / self._lr_total_updates
            new_lr = self._initial_lr * max(0.0, frac)
            for pg in self.opt["opt"].param_groups:
                pg["lr"] = new_lr
        self._lr_update_count += 1

    def prepare_batch(self, slot_view):
        return compute_gae(self.ctx, slot_view, returns_rms=self.returns_rms)

    def update(self, batch):
        self._step_lr()
        # Normalize returns on the full batch (same as PPOAlgorithm)
        if self.returns_rms is not None:
            raw_returns = batch["ret"]
            self.returns_rms.update(raw_returns)
            batch["ret"] = self.returns_rms.normalize(raw_returns)
        stats = ppo_lstm_update(self.ctx, self.policy, self.opt, batch, self.device)
        stats["learning_rate"] = self.opt["opt"].param_groups[0]["lr"]
        return stats

    def save_checkpoint(self, save_dir: str, policy: nn.Module, env_step: int) -> str:
        """Save full training state. ONNX export not supported for LSTM policies. Returns checkpoint path."""
        ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{env_step:08d}.pt")
        ckpt: dict = {
            "policy": policy.state_dict(),
            "opt": {k: v.state_dict() for k, v in self.opt.items()},
            "env_step": env_step,
            "lr_update_count": self._lr_update_count,
        }
        if self.returns_rms is not None:
            ckpt["returns_rms"] = {
                "mean": float(self.returns_rms.mean),
                "var": float(self.returns_rms.var),
                "count": float(self.returns_rms.count),
            }
        torch.save(ckpt, ckpt_path)
        logging.info("saved checkpoint to %s (ONNX export skipped for LSTM)", ckpt_path)
        return ckpt_path


def _validate_recurrence(ctx, N: int) -> int:
    recurrence = int(getattr(ctx.args, "recurrence", -1))
    if recurrence <= 0:
        recurrence = int(ctx.args.rollout)
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

    # Update obs normalizer once with the full batch, then freeze during BPTT.
    # Without this, the normalizer updates at every BPTT step, causing each
    # timestep in a sequence to be normalized with different statistics —
    # corrupting LSTM hidden state evolution.
    obs_normalizer = getattr(policy, "obs_normalizer", None)
    if obs_normalizer is not None:
        obs_normalizer.train()
        with torch.no_grad():
            obs_normalizer(data["obs"])
        obs_normalizer.eval()

    b_obs = data["obs"]
    b_actions = data["act"]
    b_logprobs = data["logp"].float()
    b_advantages = data["adv"].float()
    b_returns = data["ret"].float()
    b_values = data["val"].float()
    b_rnn_h = data.get("rnn_state_h")
    b_rnn_c = data.get("rnn_state_c")
    b_dones = data.get("done")

    if b_rnn_h is None or b_rnn_c is None:
        raise ValueError("LSTM PPO requires rnn_state_h and rnn_state_c tensors")
    if b_dones is None:
        raise ValueError("LSTM PPO requires done tensor for PackedSequence BPTT")

    N = b_obs.shape[0]
    if N == 0:
        raise ValueError("ppo_lstm_update received an empty batch")

    recurrence = _validate_recurrence(ctx, N)
    hidden = b_rnn_h.shape[-1]

    total_sequences = N // recurrence
    if total_sequences <= 0:
        raise ValueError("Not enough samples to form a recurrent minibatch")

    # Reshape to (S, T, ...) for minibatch slicing
    seq_obs = _reshape_sequences(b_obs, total_sequences, recurrence)
    seq_actions = _reshape_sequences(b_actions, total_sequences, recurrence)
    seq_logprobs = _reshape_sequences(b_logprobs, total_sequences, recurrence)
    seq_advantages = _reshape_sequences(b_advantages, total_sequences, recurrence)
    seq_returns = _reshape_sequences(b_returns, total_sequences, recurrence)
    seq_values = _reshape_sequences(b_values, total_sequences, recurrence)
    seq_rnn_h = _reshape_sequences(b_rnn_h, total_sequences, recurrence)
    seq_rnn_c = _reshape_sequences(b_rnn_c, total_sequences, recurrence)
    seq_dones = _reshape_sequences(b_dones, total_sequences, recurrence)

    seq_batch_size = ctx.args.minibatch_size // recurrence
    if seq_batch_size == 0:
        raise ValueError("minibatch_size must be at least as large as recurrence for LSTM training")

    clipfracs = []
    approx_kl = torch.tensor(0.0, device=device)
    analytical_kl = torch.tensor(0.0, device=device)

    kl_coeff = getattr(ctx.args, "kl_loss_coeff", 0.0)
    has_action_logits = "action_logits" in data
    if has_action_logits:
        from beastrand.core.model.distributions import DiagGaussianDistribution
        b_action_logits = data["action_logits"].float()
        seq_action_logits = _reshape_sequences(b_action_logits, total_sequences, recurrence)

    seq_inds = np.arange(total_sequences)
    n_mb = total_sequences / seq_batch_size
    for epoch in range(ctx.args.train_epochs):
        np.random.shuffle(seq_inds)
        for start in range(0, total_sequences, seq_batch_size):
            end = start + seq_batch_size
            mb_seq = seq_inds[start:end]
            if len(mb_seq) == 0:
                continue

            # Flatten minibatch to (S*T, ...) for PackedSequence
            S = len(mb_seq)
            mb_obs = seq_obs[mb_seq].reshape(S * recurrence, -1)
            mb_act = seq_actions[mb_seq].reshape(S * recurrence, -1)
            logprob_old_mb = seq_logprobs[mb_seq]
            adv_mb = seq_advantages[mb_seq]
            ret_mb = seq_returns[mb_seq]
            val_old_mb = seq_values[mb_seq]
            logits_old_mb = seq_action_logits[mb_seq] if has_action_logits else None

            # Concatenate h+c for PackedSequence rnn_states
            mb_rnn_states = torch.cat(
                [seq_rnn_h[mb_seq], seq_rnn_c[mb_seq]], dim=-1
            ).reshape(S * recurrence, hidden * 2)
            mb_dones = seq_dones[mb_seq].reshape(S * recurrence)

            # PackedSequence BPTT
            out = policy.evaluate_sequences(
                mb_obs, mb_act, mb_rnn_states, mb_dones, recurrence,
            )
            # Reshape flat outputs back to (S, T) for loss computation
            newlogprob = out["logp"].reshape(S, recurrence)
            entropy = out["entropy"].reshape(S, recurrence)
            newvalue = out["value"].reshape(S, recurrence)

            logratio = newlogprob - logprob_old_mb
            ratio = logratio.exp()

            # Approximate KL for monitoring
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > ctx.args.ppo_clip_range).float().mean().item())

            mb_advantages = adv_mb
            if ctx.args.normalize_adv:
                mb_advantages = normalize_advantages(mb_advantages)

            # Unbiased PPO clipping (SF-style): clip(r, 1/(1+e), 1+e)
            clip_high = 1.0 + ctx.args.ppo_clip_range
            clip_low = 1.0 / clip_high
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, clip_low, clip_high)
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
            v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ctx.args.entropy_coef * entropy_loss + v_loss * ctx.args.value_coef

            # KL penalty: analytical KL for continuous, approximate for discrete
            if kl_coeff > 0.0:
                new_logits_flat = out.get("action_logits")
                if new_logits_flat is not None and logits_old_mb is not None:
                    new_logits = new_logits_flat.reshape(S, recurrence, -1)
                    analytical_kl = DiagGaussianDistribution.kl_from_logits(
                        new_logits, logits_old_mb,
                    ).mean()
                    loss = loss + kl_coeff * analytical_kl
                else:
                    diff_approx_kl = ((ratio - 1) - logratio).mean()
                    loss = loss + kl_coeff * diff_approx_kl

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
        "analytical_kl": analytical_kl.item() if has_action_logits else 0.0,
        "clip_frac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "num_minibatches": float(n_mb),
    }
