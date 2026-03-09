"""
BufferMgr: Pre-allocate all trajectory tensors in shared PyTorch memory.

Replaces the old ShmDataSet + shm_client (multiprocessing.SharedMemory)
with a single set of shared PyTorch tensors. Workers index into them via
slot indices from a mp.Queue — no attach/detach, no extra file descriptors.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Dict, Tuple

import torch
import numpy as np

from utils.import_utils import get_object_from_path

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
    """Pre-allocate all trajectory tensors in shared PyTorch memory.

    After construction every tensor lives in shared memory (via
    ``Tensor.share_memory_()``).  Child processes forked/spawned with
    ``torch.multiprocessing`` inherit references to the same storage —
    no explicit attach/detach and zero extra file descriptors.

    Attributes:
        traj_tensors: ``{field_name: Tensor}`` with shape ``[num_traj, *field_shape]``.
        traj_buffer_queue: ``mp.Queue`` handing out free trajectory indices.
        policy_version: shared scalar tensor for learner ↔ inference version tracking.
    """

    def __init__(
        self,
        cfg,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        num_traj: int,
        T: int,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
    ) -> None:
        self.num_traj = num_traj
        self.T = T
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker

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

        # --- free-slot queue ----------------------------------------------
        self.traj_buffer_queue: mp.Queue = mp.Queue()
        for i in range(num_traj):
            self.traj_buffer_queue.put(i)

        # --- policy version (learner increments, inference server reads) --
        self.policy_version = torch.zeros(1, dtype=torch.int64)
        self.policy_version.share_memory_()

        # --- async ready flags (inference server sets, workers poll) -------
        self.ready_flags = torch.zeros(num_workers, num_envs_per_worker, dtype=torch.int32)
        self.ready_flags.share_memory_()

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
            # reverse map torch dtype to string
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
