"""Shared checkpoint / ONNX export utilities."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class ActorForExport(nn.Module):
    """Minimal actor (body + mean head) for ONNX export."""

    def __init__(self, body: nn.Module, mean_head: nn.Module):
        super().__init__()
        self.body = body
        self.mean_head = mean_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.body(obs))


def ensure_single_onnx_file(onnx_path: str) -> None:
    """Merge external data back into the .onnx protobuf if the exporter split it."""
    data_path = onnx_path + ".data"
    if not os.path.exists(data_path):
        return
    import onnx
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(model, onnx_path)
    os.remove(data_path)
