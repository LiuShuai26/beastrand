"""Shared checkpoint / ONNX export utilities."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class ActorForExport(nn.Module):
    """Generic actor exporter with optional preprocessing and discrete support."""

    def __init__(
        self,
        trunk: nn.Module,
        action_head: nn.Module,
        *,
        preprocess: nn.Module | None = None,
        discrete: bool = False,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.trunk = trunk
        self.action_head = action_head
        self.discrete = discrete

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.preprocess is not None:
            obs = self.preprocess(obs)
        logits_or_mean = self.action_head(self.trunk(obs))
        if self.discrete:
            return logits_or_mean.argmax(dim=-1, keepdim=True)
        return logits_or_mean


def ensure_single_onnx_file(onnx_path: str) -> None:
    """Merge external data back into the .onnx protobuf if the exporter split it."""
    data_path = onnx_path + ".data"
    if not os.path.exists(data_path):
        return
    import onnx
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(model, onnx_path)
    os.remove(data_path)
