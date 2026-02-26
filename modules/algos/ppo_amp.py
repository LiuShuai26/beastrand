# modules/algos/ppo_amp.py
"""
PPO-AMP algorithm: PPO with Adversarial Motion Priors.

Extends PPO with:
  - A discriminator that classifies (s_t, s_{t+1}) transition pairs as
    real (reference motion) or fake (policy-generated).
  - Style rewards computed from discriminator output (softplus).
  - Multi-group GAE: task reward (from env) + style reward (from disc).
  - Per-group value loss on dual critics.
  - Combined advantage weighting for policy loss.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from modules.amp.discriminator import AMPDiscriminator
from modules.amp.motion_buffer import AMPMotionBuffer
from modules.amp.rewards import compute_disc_loss, compute_style_reward

G = 2  # reward groups: 0 = task, 1 = style

# Joint and body order matching Brain.cpp — used by AMPMotionBuffer
JOINT_ORDER = [
    "abdomen", "neck",
    "right_shoulder", "right_elbow", "left_shoulder", "left_elbow",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
]
BODY_ORDER = ["head", "right_hand", "left_hand", "right_foot", "left_foot"]


class PPOAMPAlgorithm:
    """PPO-AMP: PPO training + discriminator training for stylized locomotion."""

    def __init__(self, ctx, policy, opt, device):
        self.ctx = ctx
        self.policy = policy
        self.opt = opt
        self.device = device
        args = ctx.args

        # AMP observation slices (configurable via args)
        self.amp_obs_slices: List[Tuple[int, int]] = list(args.amp_obs_slices)
        self.amp_obs_dim: int = args.amp_obs_dim

        # Reward weights
        self.task_reward_weight = args.task_reward_weight
        self.style_reward_weight = args.style_reward_weight

        # Motion buffer (loads keyframe data)
        self.motion_buffer = AMPMotionBuffer(
            keyframe_files=args.keyframe_file,
            joint_order=JOINT_ORDER,
            body_order=BODY_ORDER,
            device=device,
        )
        assert self.motion_buffer.obs_dim == self.amp_obs_dim, (
            f"Motion buffer obs_dim ({self.motion_buffer.obs_dim}) != "
            f"configured amp_obs_dim ({self.amp_obs_dim})"
        )

        # Discriminator
        amp_transition_dim = self.amp_obs_dim * 2
        self.discriminator = AMPDiscriminator(
            amp_transition_dim,
            hidden_dim=args.disc_hidden_dim,
            num_layers=args.disc_num_layers,
        ).to(device)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=args.disc_lr,
            weight_decay=args.disc_weight_decay,
        )

        # Thread lock: discriminator is read in ingest thread, written in main thread
        self._disc_lock = threading.Lock()

    # ------------------------------------------------------------------
    # AMP feature extraction
    # ------------------------------------------------------------------

    def _extract_amp_obs(self, obs: np.ndarray) -> np.ndarray:
        """Extract AMP features from full observations using configured slices.

        Args:
            obs: (..., obs_dim) numpy array.

        Returns:
            (..., amp_obs_dim) numpy array.
        """
        parts = [obs[..., s:e] for s, e in self.amp_obs_slices]
        return np.concatenate(parts, axis=-1)

    # ------------------------------------------------------------------
    # prepare_batch: called in Learner ingest thread per trajectory
    # ------------------------------------------------------------------

    def prepare_batch(self, view) -> None:
        """Compute AMP transitions, style rewards, multi-group GAE.

        Modifies view in-place: fills amp_transition, advantage, return_g.
        """
        arrays = view.arrays if hasattr(view, "arrays") else view
        args = self.ctx.args
        T = getattr(args, "rollout", 64)

        # 1. Extract AMP observations from raw obs [0:T+1]
        obs_np = arrays["obs"]  # (T+1, obs_dim)
        amp_obs_full = self._extract_amp_obs(obs_np)  # (T+1, amp_obs_dim)

        # 2. Build transition pairs: (s_t, s_{t+1}) for t in [0, T)
        amp_s_t = amp_obs_full[:-1]      # (T, amp_obs_dim)
        amp_s_tp1 = amp_obs_full[1:]     # (T, amp_obs_dim)
        transitions = np.concatenate([amp_s_t, amp_s_tp1], axis=-1)  # (T, 2*amp_obs_dim)
        arrays["amp_transition"][:] = transitions

        # 3. Compute style rewards via discriminator (no grad, thread-safe)
        done_np = arrays["done"]  # (T,)
        valid_mask = (done_np == 0).astype(np.float32)  # transition valid when no episode boundary

        trans_t = torch.from_numpy(transitions).float().to(self.device)
        with self._disc_lock:
            with torch.no_grad():
                normed = self.motion_buffer.normalize(trans_t)
                disc_logits = self.discriminator(normed)
                style_rewards_t = compute_style_reward(disc_logits)

        style_rewards = style_rewards_t.cpu().numpy()  # (T,)
        style_rewards *= valid_mask  # zero out invalid transitions

        # 4. Multi-group GAE
        reward_task = arrays["reward"].astype(np.float32)   # (T,)
        reward_style = style_rewards                         # (T,)
        value_g = arrays["value"].astype(np.float32)         # (T+1, 2)

        rewards_per_group = [reward_task, reward_style]
        adv_per_group = np.zeros((T, G), dtype=np.float32)
        ret_per_group = np.zeros((T, G), dtype=np.float32)

        for g in range(G):
            last_adv = 0.0
            for t in range(T - 1, -1, -1):
                nonterminal = 1.0 - float(done_np[t])
                delta = (
                    rewards_per_group[g][t]
                    + args.gamma * nonterminal * value_g[t + 1, g]
                    - value_g[t, g]
                )
                last_adv = delta + args.gamma * args.lam * nonterminal * last_adv
                adv_per_group[t, g] = last_adv
            ret_per_group[:, g] = adv_per_group[:, g] + value_g[:-1, g]

        # 5. Combine advantages with per-group normalization + weighting
        weights = [self.task_reward_weight, self.style_reward_weight]
        combined_adv = np.zeros(T, dtype=np.float32)
        for g in range(G):
            ag = adv_per_group[:, g]
            ag_std = ag.std()
            if ag_std > 1e-8:
                ag_norm = (ag - ag.mean()) / ag_std
            else:
                ag_norm = ag - ag.mean()
            combined_adv += weights[g] * ag_norm

        arrays["advantage"][:] = combined_adv
        arrays["return_g"][:] = ret_per_group

    # ------------------------------------------------------------------
    # update: called in Learner main thread on batched data
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """PPO policy update + discriminator update."""
        stats = self._ppo_update(batch)
        disc_stats = self._disc_update(batch)
        stats.update(disc_stats)
        return stats

    # ------------------------------------------------------------------
    # PPO policy + value update (with dual critics)
    # ------------------------------------------------------------------

    def _ppo_update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        args = self.ctx.args
        policy = self.policy
        optimizer = self.opt["opt"]
        device = self.device

        policy.train()

        data = _to_torch(batch, device)
        b_obs = data["obs"]
        b_actions = data["act"]
        b_logprobs = data["logp"].float()
        b_advantages = data["adv"].float()
        b_returns_g = data["ret_g"].float()    # (N, 2)
        b_values_g = data["val_g"].float()     # (N, 2)
        N = b_obs.shape[0]

        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)

        b_inds = np.arange(N)
        for epoch in range(args.train_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, N, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                inputs = {"obs": b_obs[mb_inds], "action": b_actions[mb_inds]}
                eval_out = policy.evaluate_actions(inputs)
                newlogprob = eval_out["logp"]
                entropy = eval_out["entropy"]
                newvalues_g = eval_out["value"]  # (mb, 2)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.ppo_clip_range).float().mean().item()
                    )

                # Policy loss (combined advantage)
                mb_advantages = b_advantages[mb_inds]
                if args.normalize_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.ppo_clip_range, 1 + args.ppo_clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Per-group value loss
                v_loss = torch.tensor(0.0, device=device)
                for g in range(G):
                    nv = newvalues_g[:, g]
                    v_loss_unclipped = (nv - b_returns_g[mb_inds, g]) ** 2
                    v_clipped = b_values_g[mb_inds, g] + torch.clamp(
                        nv - b_values_g[mb_inds, g],
                        -args.ppo_clip_value,
                        args.ppo_clip_value,
                    )
                    v_loss_clipped = (v_clipped - b_returns_g[mb_inds, g]) ** 2
                    v_loss = v_loss + 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.entropy_coef * entropy_loss + v_loss * args.value_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        return {
            "pi_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_frac": np.mean(clipfracs) if clipfracs else 0.0,
        }

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------

    def _disc_update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        args = self.ctx.args
        device = self.device

        data = _to_torch(batch, device)
        all_transitions = data["amp_transition"]  # (N, transition_dim)
        dones = data["done"].float()               # (N,)

        # Filter valid transitions (no episode boundary)
        valid_mask = (dones == 0)
        valid_transitions = all_transitions[valid_mask]
        n_valid = valid_transitions.shape[0]

        if n_valid == 0:
            return {
                "disc/loss": 0.0,
                "disc/grad_penalty": 0.0,
                "disc/real_accuracy": 0.0,
                "disc/fake_accuracy": 0.0,
                "disc/valid_transitions": 0.0,
            }

        disc_losses = []
        disc_gp_losses = []
        disc_real_acc = []
        disc_fake_acc = []

        disc_inds = np.arange(n_valid)
        disc_mb_size = max(1, n_valid // max(1, args.num_minibatches))

        with self._disc_lock:
            for _epoch in range(args.disc_update_epochs):
                np.random.shuffle(disc_inds)
                for start in range(0, n_valid, disc_mb_size):
                    mb_inds = disc_inds[start:start + disc_mb_size]

                    raw_fake = valid_transitions[mb_inds]
                    raw_real = self.motion_buffer.sample(len(mb_inds))
                    fake_trans = self.motion_buffer.normalize(raw_fake)
                    real_trans = self.motion_buffer.normalize(raw_real)

                    disc_real_logits = self.discriminator(real_trans)
                    disc_fake_logits = self.discriminator(fake_trans.detach())

                    disc_loss = compute_disc_loss(disc_real_logits, disc_fake_logits)
                    gp_loss = self.discriminator.compute_grad_penalty(real_trans)
                    total_loss = disc_loss + args.disc_grad_penalty_coef * gp_loss

                    self.disc_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), args.max_grad_norm
                    )
                    self.disc_optimizer.step()

                    disc_losses.append(disc_loss.item())
                    disc_gp_losses.append(gp_loss.item())
                    with torch.no_grad():
                        disc_real_acc.append(
                            (disc_real_logits > 0).float().mean().item()
                        )
                        disc_fake_acc.append(
                            (disc_fake_logits < 0).float().mean().item()
                        )

        return {
            "disc/loss": np.mean(disc_losses),
            "disc/grad_penalty": np.mean(disc_gp_losses),
            "disc/real_accuracy": np.mean(disc_real_acc),
            "disc/fake_accuracy": np.mean(disc_fake_acc),
            "disc/valid_transitions": float(n_valid),
        }

    # ------------------------------------------------------------------
    # Checkpoint saving (called by Learner on exit)
    # ------------------------------------------------------------------

    def save_checkpoint(self, save_dir: str, policy: nn.Module) -> None:
        """Save policy weights, discriminator weights, and ONNX actor."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 1. Policy state dict
        policy_path = os.path.join(save_dir, "policy.pt")
        torch.save(policy.state_dict(), policy_path)
        logging.info("saved policy to %s", policy_path)

        # 2. Discriminator state dict
        disc_path = os.path.join(save_dir, "discriminator.pt")
        torch.save(self.discriminator.state_dict(), disc_path)
        logging.info("saved discriminator to %s", disc_path)

        # 3. ONNX export (actor only: body → mean action, single file)
        try:
            actor = _ActorForExport(policy.body, policy.dist_head.mean)
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
            logging.info("saved ONNX actor to %s", onnx_path)
        except Exception:
            logging.exception("ONNX export failed")


class _ActorForExport(nn.Module):
    """Minimal actor (body + mean head) for ONNX export."""

    def __init__(self, body: nn.Module, mean_head: nn.Module):
        super().__init__()
        self.body = body
        self.mean_head = mean_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.body(obs))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_torch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            if k in ("act",):
                t = torch.from_numpy(v)
            else:
                t = torch.from_numpy(v).float()
            out[k] = t.to(device)
    return out
