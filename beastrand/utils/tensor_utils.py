"""Shared tensor utilities used across algorithm modules."""

from __future__ import annotations

from typing import Any, Dict

import torch


def to_torch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert a batch dict (numpy/torch) to torch tensors on *device*."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            # assume numpy
            if k in ("act",):
                # int or float — keep dtype
                t = torch.from_numpy(v)
            else:
                t = torch.from_numpy(v).float()
            out[k] = t.to(device)
    return out
