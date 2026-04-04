"""
BufferMgr (v2): fixed-allocation trajectory tensors in shared PyTorch memory.

Slot assignment is deterministic: each (worker, env) pair owns a fixed range
of split_depth trajectory slots, eliminating the mp.Queue coordination overhead.

  traj_idx = (worker_idx * num_envs_per_worker + env_idx) * split_depth + split_idx

Workers alternate split_idx (0, 1, 0, 1, ...) each trajectory. The Learner
releases split_flags[flat_idx, split_idx] = 0 when done processing, allowing
the worker to reuse that slot (replaces traj_buffer_queue.put()).

Inference I/O buffers are flat [W*E, *shape] to enable 1D numpy indexing
with native int32 (no astype overhead vs. 2D PyTorch fancy indexing).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from beastrand.utils.import_utils import get_object_from_path

_STR_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.uint8,
}


class BufferMgr:
    """Fixed-allocation trajectory tensors in shared PyTorch memory.

    Each (worker_idx, env_idx) pair owns split_depth contiguous trajectory
    slots. Workers cycle through splits; the Learner releases each split via
    split_flags after processing.

    Attributes:
        traj_tensors: ``{field_name: Tensor}`` with shape ``[num_traj, *field_shape]``.
        split_flags: ``[W*E, split_depth]`` int8 tensor. 0=worker-owned, 1=learner-in-use.
        policy_version: shared scalar tensor for learner ↔ inference version tracking.
        split_depth: number of trajectory slots per (worker, env) pair.
    """

    def __init__(
        self,
        cfg,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        T: int,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
        split_depth: int = 2,
    ) -> None:
        self.T = T
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.split_depth = split_depth

        W, E, D = num_workers, num_envs_per_worker, split_depth
        num_traj = W * E * D
        self.num_traj = num_traj

        # --- trajectory tensors ------------------------------------------
        data_record_path = getattr(cfg, "data_record_path", None) or cfg.args.data_record_path
        record_cls = get_object_from_path(data_record_path)
        specs = record_cls.alloc_specs(cfg, T, obs_shape, act_shape)

        self.traj_tensors: Dict[str, torch.Tensor] = {}
        for field, (shape, dtype_str) in specs.items():
            full_shape = (num_traj,) + tuple(int(x) for x in shape)
            dtype = _STR_TO_TORCH_DTYPE.get(dtype_str, torch.float32)
            t = torch.zeros(full_shape, dtype=dtype)
            t.share_memory_()
            self.traj_tensors[field] = t

        # --- policy version (learner increments, inference server reads) --
        self.policy_version = torch.zeros(1, dtype=torch.int64)
        self.policy_version.share_memory_()

        # --- split flags [W*E, split_depth]: 0=worker-owned, 1=learner-in-use ---
        # Replaces traj_buffer_queue. Learner sets to 0 when done processing a slot;
        # worker spin-waits on this flag before reusing the slot (backpressure).
        self.split_flags = torch.zeros(W * E, D, dtype=torch.int8)
        self.split_flags.share_memory_()

        # --- async ready flags [W*E] flat (inference server sets, workers poll) ---
        self.ready_flags = torch.zeros(W * E, dtype=torch.int32)
        self.ready_flags.share_memory_()

        # --- LSTM live state buffers [W*E, hidden] flat ---
        # Flat layout matches the inference I/O buffers for consistent 1D indexing.
        self.rnn_state_live_h: Optional[torch.Tensor] = None
        self.rnn_state_live_c: Optional[torch.Tensor] = None
        if "rnn_state_h" in self.traj_tensors:
            hidden = self.traj_tensors["rnn_state_h"].shape[2]  # [num_traj, T+1, hidden]
            self.rnn_state_live_h = torch.zeros(W * E, hidden)
            self.rnn_state_live_h.share_memory_()
            self.rnn_state_live_c = torch.zeros(W * E, hidden)
            self.rnn_state_live_c.share_memory_()

        # --- Inference I/O buffers [W*E, *shape] flat ---
        # Flat layout (vs. old [W, E, *shape]) enables 1D numpy int32 indexing
        # in IS._gather_obs, avoiding astype(int64) and PyTorch 2D fancy indexing.
        WE = W * E
        self.infer_obs = torch.zeros(WE, *obs_shape)
        self.infer_obs.share_memory_()
        self.infer_act = torch.zeros(WE, *act_shape)
        self.infer_act.share_memory_()
        self.infer_logp = torch.zeros(WE)
        self.infer_logp.share_memory_()
        # Match value shape from traj_tensors (may have trailing dims beyond scalar).
        val_trailing = self.traj_tensors["value"].shape[2:]
        self.infer_val = torch.zeros(WE, *val_trailing)
        self.infer_val.share_memory_()
        # mask used only for LSTM (non-LSTM code ignores it)
        self.infer_mask = torch.zeros(WE)
        self.infer_mask.share_memory_()
        # action_logits: only allocated when traj_tensors contains the field
        # (continuous action spaces with kl_loss_coeff > 0)
        self.infer_action_logits: Optional[torch.Tensor] = None
        if "action_logits" in self.traj_tensors:
            logits_dim = self.traj_tensors["action_logits"].shape[2]  # [num_traj, T, dim]
            self.infer_action_logits = torch.zeros(WE, logits_dim)
            self.infer_action_logits.share_memory_()

        # --- Extra agent inference buffers (multi-agent / self-play) ------
        # Agent 0 uses the buffers above (full: obs, act, logp, val, etc.).
        # Agents 1..N-1 use one of two paths:
        #   - latest_policy_infer: served by the training-policy IS (latest self-play)
        #   - snapshot_infer: served by dedicated snapshot opponent IS processes
        # When num_agents=1 (default), both lists are empty → zero overhead.
        _args = getattr(cfg, "args", cfg)
        self.num_agents: int = getattr(_args, "num_agents", 1)
        self.latest_policy_infer: List[Dict[str, torch.Tensor]] = []
        self.snapshot_infer: List[Dict[str, torch.Tensor]] = []
        for _ in range(1, self.num_agents):
            self.latest_policy_infer.append({
                "obs": torch.zeros(WE, *obs_shape).share_memory_(),
                "act": torch.zeros(WE, *act_shape).share_memory_(),
                "ready_flags": torch.zeros(WE, dtype=torch.int32).share_memory_(),
            })
            self.snapshot_infer.append({
                "obs": torch.zeros(WE, *obs_shape).share_memory_(),
                "act": torch.zeros(WE, *act_shape).share_memory_(),
                "ready_flags": torch.zeros(WE, dtype=torch.int32).share_memory_(),
            })

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def schema(self) -> Dict[str, Dict]:
        """Return ``{field: {shape: tuple, dtype: str}}`` (per-slot, without the
        leading ``num_traj`` dimension) so that DataRecord.build_batch and other
        consumers know the layout."""
        out: Dict[str, Dict] = {}
        for field, t in self.traj_tensors.items():
            slot_shape = tuple(t.shape[1:])  # drop num_traj dim
            dtype_str = _torch_dtype_to_str(t.dtype)
            out[field] = {"shape": slot_shape, "dtype": dtype_str}
        return out

    def slot_as_numpy(self, slot_idx: int) -> Dict[str, np.ndarray]:
        """Return numpy views into a single slot (zero-copy)."""
        out: Dict[str, np.ndarray] = {}
        for field, t in self.traj_tensors.items():
            out[field] = t[slot_idx].numpy()
        return out


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    for s, d in _STR_TO_TORCH_DTYPE.items():
        if d == dtype:
            return s
    return "float32"
