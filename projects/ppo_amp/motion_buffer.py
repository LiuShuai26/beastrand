"""
Motion clip buffer for AMP training.

Loads keyframe JSON files and provides random sampling of (s_t, s_{t+1})
transition pairs for discriminator training.

Per-frame AMP state features (matching Brain.cpp observation layout):
    [pelvis_y(1), joint_sin_cos(24), joint_angvel(12),
     body_pos_pelvis_frame(10), phase(1)] = 48 dims

When the environment observation does not include a phase channel
(e.g. VAEBrain, 47-dim AMP features), pass ``include_phase=False``
to drop the phase column so the buffer matches the policy's feature space.

When ``include_angvel=False``, joint angular velocities are omitted,
giving the original 36/35-dim layout.

- Joints are stored as sin/cos pairs (not raw angles) to match the raw
  observation format and avoid lossy atan2 roundtrips.
- Joint angular velocities are computed via finite differences between
  consecutive keyframes: omega = (angle[t+1] - angle[t]) / dt.
- Body positions are rotated into the pelvis local frame using the same
  transform as Brain.cpp:  dx*cos(-a) - dy*sin(-a), dx*sin(-a) + dy*cos(-a).
- pelvis_x is excluded (horizontal translation invariance).
- Phase is a scalar in [0,1] indicating position within the clip.
  Cyclic clips: phase = frame_idx / num_frames (wraps).
  Aperiodic clips: phase = frame_idx / (num_frames - 1) (clamped).
"""

import json
import logging
import math

import numpy as np
import torch


class AMPMotionBuffer:
    """Buffer of reference motion transition pairs for AMP discriminator.

    Pre-computes per-frame AMP states from keyframe data, applying the
    same coordinate transforms as Brain.cpp, then builds consecutive
    (s_t, s_{t+1}) transition pairs.  For cyclic motions the last frame
    wraps to the first so no transitions are lost.

    Args:
        keyframe_files: Single path or list of paths to keyframe JSON files.
        joint_order:    Joint names in the order they appear in the observation.
                        Must match Brain.cpp JOINT_TAGS.
        body_order:     Key body names in the order they appear in the observation.
                        Must match HumanoidConfig.h KEY_BODY_TAGS.
        device:         Torch device for pre-computed tensors.
        include_phase:  Include phase channel in AMP features.
        include_angvel: Include joint angular velocities in AMP features.

    Attributes:
        obs_dim:        Dimensionality of a single-frame AMP state.
        transition_dim: Dimensionality of a transition pair (obs_dim * 2).
    """

    def __init__(
        self,
        keyframe_files,
        joint_order,
        body_order,
        device="cpu",
        include_phase=True,
        include_angvel=True,
    ):
        if isinstance(keyframe_files, str):
            keyframe_files = [keyframe_files]

        self.device = device
        self.joint_order = list(joint_order)
        self.body_order = list(body_order)
        self.num_joints = len(self.joint_order)
        self.num_bodies = len(self.body_order)
        self.include_phase = include_phase
        self.include_angvel = include_angvel

        # Compute obs_dim:
        # pelvis_y(1) + sin/cos(num_joints*2) + [angvel(num_joints)] + body_pos(num_bodies*2) + [phase(1)]
        base_dim = 1 + self.num_joints * 2 + self.num_bodies * 2
        if include_angvel:
            base_dim += self.num_joints
        self.obs_dim = base_dim + 1 if include_phase else base_dim
        self.transition_dim = self.obs_dim * 2

        clips = []  # list of (F, obs_dim) arrays, one per file
        clip_cyclic = []  # whether each clip is cyclic

        for kf_file in keyframe_files:
            with open(kf_file) as f:
                data = json.load(f)
            keyframes = data["keyframes"]
            is_cyclic = data.get("is_cyclic", True)
            dt = data.get("dt", 1.0 / 60.0)  # time step between frames
            clip_cyclic.append(is_cyclic)

            num_frames = len(keyframes)

            # 1. Extract raw joint angles for angular velocity computation
            raw_angles = np.zeros((num_frames, self.num_joints), dtype=np.float32)
            for i, kf in enumerate(keyframes):
                for j, jname in enumerate(self.joint_order):
                    raw_angles[i, j] = kf.get(jname, 0.0)

            # 2. Compute angular velocities via finite differences
            if include_angvel:
                angvel = np.zeros((num_frames, self.num_joints), dtype=np.float32)
                for i in range(num_frames):
                    if is_cyclic:
                        i_next = (i + 1) % num_frames
                    else:
                        i_next = min(i + 1, num_frames - 1)
                    angvel[i] = (raw_angles[i_next] - raw_angles[i]) / dt

            # 3. Build per-frame AMP features
            clip = np.zeros((num_frames, self.obs_dim), dtype=np.float32)
            for i, kf in enumerate(keyframes):
                feat = self._keyframe_to_amp_base(kf)
                if include_angvel:
                    feat = np.concatenate([feat[:1 + self.num_joints * 2],
                                           angvel[i],
                                           feat[1 + self.num_joints * 2:]])
                if include_phase:
                    if num_frames <= 1:
                        phase = 0.0
                    elif is_cyclic:
                        phase = i / num_frames
                    else:
                        phase = i / (num_frames - 1)
                    clip[i, :-1] = feat
                    clip[i, -1] = phase
                else:
                    clip[i] = feat
            clips.append(clip)

        # Build transition pairs per clip, then concatenate
        all_transitions = []
        total_frames = 0
        for clip, is_cyclic in zip(clips, clip_cyclic):
            total_frames += len(clip)
            if is_cyclic:
                # Cyclic: last frame wraps to first (N transitions from N frames)
                s_tp1 = np.roll(clip, -1, axis=0)
                transitions = np.concatenate([clip, s_tp1], axis=-1)
            else:
                # Aperiodic: consecutive pairs only (N-1 transitions from N frames)
                transitions = np.concatenate(
                    [clip[:-1], clip[1:]], axis=-1
                )
            all_transitions.append(transitions)

        all_transitions = np.concatenate(all_transitions, axis=0)
        self.num_transitions = len(all_transitions)
        self.amp_transitions = torch.tensor(all_transitions, device=device)

        # Keep single-frame obs for debugging
        all_frames = np.concatenate(clips, axis=0)
        self.amp_obs = torch.tensor(all_frames, device=device)
        self.num_frames = total_frames

        # Normalization stats computed from reference motion data
        self.obs_mean = self.amp_obs.mean(dim=0)
        raw_std = self.amp_obs.std(dim=0)
        self.obs_std = raw_std.clamp(min=0.1)

        logging.info(
            "AMPMotionBuffer: %d frames, %d transitions, "
            "obs_dim=%d, transition_dim=%d, angvel=%s, phase=%s",
            total_frames, self.num_transitions, self.obs_dim, self.transition_dim,
            include_angvel, include_phase,
        )

    def _keyframe_to_amp_base(self, kf):
        """Convert one keyframe dict to base AMP state (no angvel, no phase).

        Returns array: [pelvis_y, sin(j0), cos(j0), ..., body0_lx, body0_ly, ...]
        Angular velocity and phase are inserted/appended by the caller.
        """
        feat = []

        # 1. pelvis_y (height)
        feat.append(kf.get("pelvis_y", 0.0))

        # 2. Joint angles -> sin/cos pairs (same order as Brain.cpp)
        for jname in self.joint_order:
            angle = kf.get(jname, 0.0)
            feat.append(math.sin(angle))
            feat.append(math.cos(angle))

        # 3. Body positions rotated into pelvis local frame
        pelvis_angle = kf.get("pelvis_angle", 0.0)
        cos_a = math.cos(-pelvis_angle)
        sin_a = math.sin(-pelvis_angle)

        for bname in self.body_order:
            dx = kf.get(f"{bname}_x", 0.0)
            dy = kf.get(f"{bname}_y", 0.0)
            local_x = dx * cos_a - dy * sin_a
            local_y = dx * sin_a + dy * cos_a
            feat.append(local_x)
            feat.append(local_y)

        return np.array(feat, dtype=np.float32)

    def normalize(self, amp_obs):
        """Normalize AMP observations using reference motion statistics.

        Applied to both real and fake inputs before the discriminator so
        scale differences don't provide a trivial classification signal.
        For transition pairs, normalizes each half independently.
        Output is clamped to [-5, 5] to prevent extreme values when
        policy states are far from the reference distribution.

        Args:
            amp_obs: (B, obs_dim) single frames or (B, transition_dim) pairs.
        """
        if amp_obs.shape[-1] == self.transition_dim:
            s_t = (amp_obs[..., :self.obs_dim] - self.obs_mean) / self.obs_std
            s_tp1 = (amp_obs[..., self.obs_dim:] - self.obs_mean) / self.obs_std
            return torch.clamp(torch.cat([s_t, s_tp1], dim=-1), -5.0, 5.0)
        return torch.clamp((amp_obs - self.obs_mean) / self.obs_std, -5.0, 5.0)

    def sample(self, batch_size):
        """Sample random transition pairs.

        Returns:
            Tensor of shape (batch_size, transition_dim) on self.device.
        """
        indices = torch.randint(0, self.num_transitions, (batch_size,))
        return self.amp_transitions[indices]
