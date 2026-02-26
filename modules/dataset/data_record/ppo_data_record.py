# modules/dataset/ppo_data_record.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from .base_record import DataRecordBase

class PPODataRecord(DataRecordBase):
    @staticmethod
    def alloc_specs(ctx, T: int, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...]):
        # Discrete action example: action shape [T, 1] int64; for Box, use float32 [T, *act_shape]
        return {
            "obs":       ((T+1, *obs_shape), "float32"),
            "action":    ((T, *act_shape),          "float32"),   # or (T, *act_shape) float32 for Box
            "reward": ((T,), "float32"),
            "terminated": ((T,), "uint8"),
            "truncated": ((T,), "uint8"),
            "done": ((T,), "uint8"),
            "log_prob":  ((T,),            "float32"),
            "value":     ((T+1,),            "float32"),
            "advantage": ((T,), "float32"),
            "return": ((T,), "float32"),
            "model_version": ((T,), "int32"),
        }

    @staticmethod
    def build_batch(ctx, view: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "obs":  view["obs"].astype(np.float32),
            "act":  view["action"],  # keep dtype (int64 for discrete; float32 for Box)
            "logp": view["log_prob"].astype(np.float32),
            "adv":  view["advantage"].astype(np.float32),
            "ret":  view["return"].astype(np.float32),
            "val":  view["value"].astype(np.float32),
        }
