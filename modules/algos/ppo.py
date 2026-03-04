# algos/ppo.py
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

    def prepare_batch(self, slot_view):
        # PPO-specific prep: bootstrap + GAE
        return compute_gae(self.ctx, slot_view)

    def update(self, batch):
        return ppo_update(self.ctx, self.policy, self.opt, batch, self.device)

    def save_checkpoint(self, save_dir: str, policy: nn.Module) -> None:
        """Save policy weights and ONNX actor."""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Policy state dict
        policy_path = os.path.join(save_dir, "policy.pt")
        torch.save(policy.state_dict(), policy_path)
        logging.info("saved policy to %s", policy_path)

        # 2. ONNX export (actor only: body → mean action, single file)
        try:
            actor = ActorForExport(policy.body, policy.dist_head.mean)
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


def compute_gae(ctx, view) -> None:
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
    lost due to Beast's auto-reset inside Habitat::Step).
    """
    T = ctx.args.rollout
    gamma = ctx.args.gamma
    arrays = view
    has_truncated = "truncated" in arrays
    adv = np.zeros_like(arrays["advantage"], dtype=np.float32)
    last_adv = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - float(arrays["done"][t])
        r_t = float(arrays["reward"][t])
        if has_truncated and arrays["truncated"][t]:
            r_t += gamma * float(arrays["value"][t])
        delta = r_t + gamma * nonterminal * float(arrays["value"][t + 1]) - float(arrays["value"][t])
        last_adv = delta + gamma * ctx.args.lam * nonterminal * last_adv
        adv[t] = last_adv
    arrays["advantage"] = adv
    arrays["return"] = adv + arrays["value"][:-1].astype(np.float32)


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
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > ctx.args.ppo_clip_range).float().mean().item())

            mb_advantages = b_advantages[mb_inds]
            if ctx.args.normalize_adv:
                mb_advantages = normalize_advantages(mb_advantages)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - ctx.args.ppo_clip_range, 1 + ctx.args.ppo_clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -ctx.args.ppo_clip_value,
                ctx.args.ppo_clip_value,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

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
        "clip_frac": np.mean(clipfracs),
        "num_minibatches": float(n_mb),
    }


