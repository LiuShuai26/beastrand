from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .ppo_data_record import PPODataRecord


class PPOLSTMDataRecord(PPODataRecord):
    @staticmethod
    def alloc_specs(ctx, T: int, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...]):
        specs = dict(PPODataRecord.alloc_specs(ctx, T, obs_shape, act_shape))
        hidden = int(getattr(ctx.args, "lstm_hidden_size", 0))
        if hidden <= 0:
            raise ValueError("PPOLSTMDataRecord requires ctx.args.lstm_hidden_size > 0")
        specs.update({
            "rnn_state_h": ((T + 1, hidden), "float32"),
            "rnn_state_c": ((T + 1, hidden), "float32"),
            "mask": ((T + 1,), "float32"),
        })
        return specs

    @staticmethod
    def build_batch(ctx, view: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch = PPODataRecord.build_batch(ctx, view)
        steps = view["log_prob"].shape[0] if "log_prob" in view else view["action"].shape[0]
        batch["rnn_state_h"] = view["rnn_state_h"][:steps].astype(np.float32)
        batch["rnn_state_c"] = view["rnn_state_c"][:steps].astype(np.float32)
        if "mask" in view:
            batch["mask"] = view["mask"][:steps].astype(np.float32)
        return batch
