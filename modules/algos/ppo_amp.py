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
from utils.tensor_utils import to_torch

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
        """Phase 1: extract AMP transition pairs (numpy only, no GPU).

        Writes amp_transition into the view.  Style rewards, GAE, and
        advantage are deferred to :meth:`prepare_batch_finalize` which
        runs a single batched discriminator forward for many trajectories.
        """
        arrays = view
        obs_np = arrays["obs"]  # (T+1, obs_dim)
        amp_obs_full = self._extract_amp_obs(obs_np)  # (T+1, amp_obs_dim)
        transitions = np.concatenate(
            [amp_obs_full[:-1], amp_obs_full[1:]], axis=-1,
        )  # (T, 2*amp_obs_dim)
        arrays["amp_transition"][:] = transitions

    def prepare_batch_finalize(self, views: List[Dict[str, np.ndarray]]) -> None:
        """Phase 2: batched discriminator forward + per-trajectory GAE.

        Called once per batch on *all* accumulated trajectory views after
        :meth:`prepare_batch` has been called on each view individually.
        This replaces N serial GPU forward passes with a single batched
        forward, yielding 10-16x speedup on the discriminator hot path.

        Args:
            views: list of per-trajectory numpy view dicts.  Each view
                   must already have ``amp_transition`` populated by
                   :meth:`prepare_batch`.  ``value`` must still contain
                   the bootstrap row at index T (shared-memory view,
                   not yet copied into BatchBuffer).
        """
        args = self.ctx.args
        T = args.rollout
        n_traj = len(views)
        if n_traj == 0:
            return

        # -- 1. Gather all transitions into one (n_traj*T, dim) array -----
        all_trans = np.stack(
            [v["amp_transition"][:T] for v in views], axis=0,
        )  # (n_traj, T, transition_dim)
        all_trans_flat = all_trans.reshape(-1, all_trans.shape[-1])

        # -- 2. Single batched discriminator forward ----------------------
        trans_t = torch.from_numpy(all_trans_flat).float().to(self.device)
        with self._disc_lock:
            with torch.no_grad():
                normed = self.motion_buffer.normalize(trans_t)
                disc_logits = self.discriminator(normed)
                style_rewards_all = compute_style_reward(disc_logits)
        style_rewards_np = style_rewards_all.cpu().numpy().reshape(n_traj, T)

        # -- 3. Per-trajectory GAE ----------------------------------------
        gamma = args.gamma
        lam = args.lam
        weights = [self.task_reward_weight, self.style_reward_weight]

        for i, view in enumerate(views):
            done_np = view["done"]  # (T,)
            terminated_np = view["terminated"]  # (T,)
            valid_mask = (done_np == 0).astype(np.float32)
            style_rewards = style_rewards_np[i] * valid_mask

            reward_task = view["reward"].astype(np.float32)
            value_g = view["value"].astype(np.float32)  # (T+1, 2)

            rewards_per_group = [reward_task, style_rewards]
            adv_per_group = np.zeros((T, G), dtype=np.float32)
            ret_per_group = np.zeros((T, G), dtype=np.float32)

            for g in range(G):
                last_adv = 0.0
                for t in range(T - 1, -1, -1):
                    nonterminal = 1.0 - float(terminated_np[t])
                    delta = (
                        rewards_per_group[g][t]
                        + gamma * nonterminal * value_g[t + 1, g]
                        - value_g[t, g]
                    )
                    last_adv = delta + gamma * lam * nonterminal * last_adv
                    adv_per_group[t, g] = last_adv
                ret_per_group[:, g] = adv_per_group[:, g] + value_g[:-1, g]

            combined_adv = np.zeros(T, dtype=np.float32)
            for g in range(G):
                ag = adv_per_group[:, g]
                ag_std = ag.std()
                if ag_std > 1e-8:
                    ag_norm = (ag - ag.mean()) / ag_std
                else:
                    ag_norm = ag - ag.mean()
                combined_adv += weights[g] * ag_norm

            view["advantage"][:] = combined_adv
            view["return_g"][:] = ret_per_group

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

        data = to_torch(batch, device)
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

        data = to_torch(batch, device)
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

                    # Add noise to prevent discriminator saturation
                    if args.disc_noise_std > 0:
                        real_trans = real_trans + args.disc_noise_std * torch.randn_like(real_trans)
                        fake_trans = fake_trans + args.disc_noise_std * torch.randn_like(fake_trans)

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

        # 2. Discriminator + optimizer state
        disc_path = os.path.join(save_dir, "discriminator.pt")
        torch.save({
            "discriminator": self.discriminator.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
        }, disc_path)
        logging.info("saved discriminator to %s", disc_path)

        # 3. Motion buffer normalization stats
        stats_path = os.path.join(save_dir, "amp_stats.pt")
        torch.save({
            "obs_mean": self.motion_buffer.obs_mean,
            "obs_std": self.motion_buffer.obs_std,
        }, stats_path)
        logging.info("saved AMP stats to %s", stats_path)

        # 4. ONNX export (actor only: body → mean action, single file)
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
            _ensure_single_onnx_file(onnx_path)
            logging.info("saved ONNX actor to %s", onnx_path)
        except Exception:
            logging.exception("ONNX export failed")


    def load_checkpoint(self, save_dir: str, policy: nn.Module) -> None:
        """Load policy weights, discriminator weights, and optimizer state."""
        import os

        # 1. Policy
        policy_path = os.path.join(save_dir, "policy.pt")
        if os.path.exists(policy_path):
            policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            logging.info("loaded policy from %s", policy_path)

        # 2. Discriminator + optimizer
        disc_path = os.path.join(save_dir, "discriminator.pt")
        if os.path.exists(disc_path):
            checkpoint = torch.load(disc_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "discriminator" in checkpoint:
                self.discriminator.load_state_dict(checkpoint["discriminator"])
                if "disc_optimizer" in checkpoint:
                    self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
            else:
                # Legacy format: bare state_dict
                self.discriminator.load_state_dict(checkpoint)
            logging.info("loaded discriminator from %s", disc_path)


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

def _ensure_single_onnx_file(onnx_path: str) -> None:
    """Merge external data back into the .onnx protobuf if the exporter split it."""
    import os as _os
    data_path = onnx_path + ".data"
    if not _os.path.exists(data_path):
        return
    import onnx
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(model, onnx_path)
    _os.remove(data_path)


