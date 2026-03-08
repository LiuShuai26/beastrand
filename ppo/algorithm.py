# ppo/algorithm.py
"""
Minimal PPO update utilities.

- PPOConfig: hyperparams
- ppo_update(): one or more epochs of minibatch updates
- Advantage normalization inside the function (or pass pre-normalized)

Expected batch dict keys (NumPy or torch tensors; we convert to torch):
    obs:   [N, *obs_shape] float32
    act:   [N, 1] int64 for discrete OR [N, act_dim] float32 for box
    logp:  [N] old log-prob
    adv:   [N] advantage (will be normalized unless `normalize_adv=False`)
    ret:   [N] return/target value

Model API:
    - policy: ActorCritic with .evaluate_actions({"obs", "action"}) and .value({"obs"})
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.checkpoint_utils import ActorForExport, ensure_single_onnx_file
from utils.tensor_utils import to_torch  # noqa: F401  (re-exported for ppo_lstm)


class PPOAlgorithm:
    def __init__(self, ctx, policy, opt, device):
        self.ctx = ctx
        self.policy = policy
        self.opt = opt
        self.device = device

        self.returns_rms = None
        if getattr(ctx.args, "normalize_returns", False):
            from core.running_mean_std import RunningMeanStd
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
        # PPO-specific prep: bootstrap + GAE
        return compute_gae(self.ctx, slot_view, returns_rms=self.returns_rms)

    def update(self, batch):
        self._step_lr()
        # Normalize returns on the full batch (not per-trajectory)
        if self.returns_rms is not None:
            raw_returns = batch["ret"]
            self.returns_rms.update(raw_returns)
            batch["ret"] = self.returns_rms.normalize(raw_returns)
        stats = ppo_update(self.ctx, self.policy, self.opt, batch, self.device)
        stats["learning_rate"] = self.opt["opt"].param_groups[0]["lr"]
        return stats

    def save_checkpoint(self, save_dir: str, policy: nn.Module) -> None:
        """Save policy weights and ONNX actor."""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Policy state dict
        policy_path = os.path.join(save_dir, "policy.pt")
        torch.save(policy.state_dict(), policy_path)
        logging.info("saved policy to %s", policy_path)

        # 2. ONNX export (actor only: body → deterministic action, single file)
        try:
            if hasattr(policy.dist_head, "mean"):
                # Continuous: body → mean linear layer
                action_head = policy.dist_head.mean
            elif hasattr(policy.dist_head, "logits"):
                # Discrete: body → logits linear layer
                action_head = policy.dist_head.logits
            else:
                raise AttributeError("Unknown dist_head type for ONNX export")
            actor = ActorForExport(policy.body, action_head)
            actor.eval()
            obs_dim = policy.obs_dim
            dummy = torch.zeros(1, obs_dim, device=self.device)
            onnx_path = os.path.join(save_dir, "actor.onnx")
            torch.onnx.export(
                actor,
                dummy,
                onnx_path,
                input_names=["obs"],
                output_names=["action_mean"],
                dynamic_axes={"obs": {0: "batch"}, "action_mean": {0: "batch"}},
            )
            ensure_single_onnx_file(onnx_path)
            logging.info("saved ONNX actor to %s", onnx_path)
        except Exception:
            logging.exception("ONNX export failed")




def normalize_advantages(adv: torch.Tensor) -> torch.Tensor:
    return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def compute_gae(ctx, view, returns_rms=None) -> None:
    """
    Compute GAE advantages and returns, writing results into the view dict.

    view: Dict[str, np.ndarray] with keys: reward, done, value, advantage, return

    Uses ``done`` (not ``terminated``) for the nonterminal mask because
    trajectories use auto-reset: after any done (terminated *or* truncated),
    obs[t+1] belongs to a NEW episode.  Bootstrapping across that boundary
    would leak the next episode's value into the current one.

    For **truncated** episodes (time-limit, agent still alive), we apply
    SB3-style reward correction: ``r_t += gamma * V(obs_t)`` so the agent
    learns that truncation is not a catastrophic failure.  ``V(obs_t)`` is
    an approximation of ``V(s_terminal)`` (the true terminal observation is
    lost due to auto-reset).

    If ``returns_rms`` is provided (SF-style normalize_returns), values are
    denormalized before GAE.  Returns normalization happens later on the
    full batch in ``PPOAlgorithm.update()``.
    """
    T = ctx.args.rollout
    gamma = ctx.args.gamma
    arrays = view
    has_truncated = "truncated" in arrays

    # Values from the critic — denormalize if normalize_returns is active
    values = arrays["value"][:T + 1].astype(np.float32)
    if returns_rms is not None:
        values = returns_rms.denormalize(values)

    adv = np.zeros_like(arrays["advantage"], dtype=np.float32)
    last_adv = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - float(arrays["done"][t])
        r_t = float(arrays["reward"][t])
        if has_truncated and arrays["truncated"][t]:
            r_t += gamma * float(values[t])
        delta = r_t + gamma * nonterminal * float(values[t + 1]) - float(values[t])
        last_adv = delta + gamma * ctx.args.lam * nonterminal * last_adv
        adv[t] = last_adv

    arrays["advantage"] = adv
    returns = adv + values[:-1]

    arrays["return"] = returns


def ppo_update(
        ctx,
        policy: nn.Module,
        opt: Dict[str, optim.Optimizer],
        batch: Dict[str, Any],
        device: torch.device,
) -> Dict[str, float]:
    """
    Runs PPO for ctx.args.epochs over minibatches sampled from `batch`.
    Expects batch keys: obs, act, logp (old), adv, ret, and optionally valu (old V(s)).

    NOTE: Uses standard PPO clipped ratio (not V-trace). This works well when
    policy lag is small, which is typical for our worker/learner
    ratio. If scaling to hundreds of workers or very slow envs where lag grows
    large, consider adding V-trace off-policy correction to fix
    value target bias that PPO clip alone cannot handle. See Sample Factory's
    ``--with_vtrace`` for reference.
    """
    policy.train()

    optimizer = opt["opt"]

    data = to_torch(batch, device)

    b_obs = data["obs"]
    b_actions = data["act"]
    b_logprobs = data["logp"].float()
    b_advantages = data["adv"].float()
    b_returns = data["ret"].float()
    b_values = data["val"].float()
    N = b_obs.shape[0]
    if N == 0:
        raise ValueError("ppo_update received an empty batch")

    clipfracs = []
    approx_kl = torch.tensor(0.0, device=device)

    n_mb = N / ctx.args.minibatch_size
    b_inds = np.arange(b_obs.shape[0])
    for epoch in range(ctx.args.train_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, b_obs.shape[0], ctx.args.minibatch_size):
            end = start + ctx.args.minibatch_size
            mb_inds = b_inds[start:end]

            inputs = {"obs": b_obs[mb_inds], "action": b_actions[mb_inds]}

            eval_out = policy.evaluate_actions(inputs)
            newlogprob, entropy, newvalue = eval_out["logp"], eval_out["entropy"], eval_out["value"]
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = torch.clamp(logratio.exp(), 0.05, 20.0)

            # Differentiable KL for optional penalty term
            approx_kl = ((ratio - 1) - logratio).mean()
            with torch.no_grad():
                clipfracs.append(((ratio - 1.0).abs() > ctx.args.ppo_clip_range).float().mean().item())

            mb_advantages = b_advantages[mb_inds]
            if ctx.args.normalize_adv:
                mb_advantages = normalize_advantages(mb_advantages)

            # Unbiased PPO clipping (SF-style): clip(r, 1/(1+e), 1+e)
            clip_high = 1.0 + ctx.args.ppo_clip_range
            clip_low = 1.0 / clip_high
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, clip_low, clip_high)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -ctx.args.ppo_clip_value,
                ctx.args.ppo_clip_value,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ctx.args.entropy_coef * entropy_loss + v_loss * ctx.args.value_coef

            kl_coeff = getattr(ctx.args, "kl_loss_coeff", 0.0)
            if kl_coeff > 0.0:
                loss = loss + kl_coeff * approx_kl

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
        "clip_frac": np.mean(clipfracs),
        "num_minibatches": float(n_mb),
    }


