# modules/dataset/data_record/ppo_amp_data_record.py
"""
Data record for PPO-AMP: extends PPO layout with dual value heads,
per-group returns, and AMP transition pairs for discriminator training.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from .base_record import DataRecordBase


class PPOAMPDataRecord(DataRecordBase):

    @staticmethod
    def alloc_specs(ctx, T: int, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...]):
        amp_obs_dim = ctx.args.amp_obs_dim
        amp_transition_dim = amp_obs_dim * 2
        return {
            "obs":            ((T + 1, *obs_shape), "float32"),
            "action":         ((T, *act_shape), "float32"),
            "reward":         ((T,), "float32"),            # task reward (from env)
            "terminated":     ((T,), "uint8"),
            "truncated":      ((T,), "uint8"),
            "done":           ((T,), "uint8"),
            "log_prob":       ((T,), "float32"),
            "value":          ((T + 1, 2), "float32"),      # dual critics [task, style]
            "advantage":      ((T,), "float32"),            # combined weighted advantage
            "return_g":       ((T, 2), "float32"),          # per-group returns
            "amp_transition": ((T, amp_transition_dim), "float32"),  # (s_t, s_{t+1}) pairs
            "model_version":  ((T,), "int32"),
        }

    @staticmethod
    def build_batch(ctx, view: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "obs":            view["obs"].astype(np.float32),
            "act":            view["action"],
            "logp":           view["log_prob"].astype(np.float32),
            "adv":            view["advantage"].astype(np.float32),
            "ret_g":          view["return_g"].astype(np.float32),
            "val_g":          view["value"].astype(np.float32),
            "amp_transition": view["amp_transition"].astype(np.float32),
            "done":           view["done"],
        }
