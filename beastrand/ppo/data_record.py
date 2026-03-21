# ppo/data_record.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from beastrand.core.base_record import DataRecordBase

class PPODataRecord(DataRecordBase):
    @staticmethod
    def alloc_specs(ctx, T: int, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...]):
        act_dtype = "int64" if getattr(ctx, "act_kind", None) == "discrete" else "float32"
        specs = {
            "obs":       ((T+1, *obs_shape), "float32"),
            "action":    ((T, *act_shape),          act_dtype),
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
        # Store distribution parameters for analytical KL (continuous only)
        if getattr(ctx, "act_kind", None) == "box":
            act_dim = act_shape[0]
            specs["action_logits"] = ((T, act_dim * 2), "float32")
        return specs

    @staticmethod
    def build_batch(ctx, view: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch = {
            "obs":  view["obs"].astype(np.float32),
            "act":  view["action"],  # keep dtype (int64 for discrete; float32 for Box)
            "logp": view["log_prob"].astype(np.float32),
            "adv":  view["advantage"].astype(np.float32),
            "ret":  view["return"].astype(np.float32),
            "val":  view["value"].astype(np.float32),
        }
        if "action_logits" in view:
            batch["action_logits"] = view["action_logits"].astype(np.float32)
        return batch
