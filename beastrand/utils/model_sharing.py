"""
Lightweight weight-sharing between Learner and InferenceServer.

Replaces the old ParamServer process + mp.Queue RPC approach with
shared-memory state_dict + Lock + version counter.

Usage:
    # Learner side
    param_server = ParameterServer(policy, device, buffer_mgr.policy_version)
    ...
    param_server.update(policy)  # after optimizer.step()

    # InferenceServer side
    param_client = ParameterClient(param_server)
    ...
    param_client.ensure_updated(local_policy)  # before batched inference
"""
from __future__ import annotations

import multiprocessing as mp
from collections import OrderedDict
from typing import Mapping

import torch


class ParameterServer:
    """Created by Learner. Holds a shared-memory copy of policy weights.

    Attributes:
        shared_state: OrderedDict of shared-memory tensors (same keys as policy.state_dict()).
        policy_version: Shared int64 tensor (from BufferMgr). Bumped on every update().
        lock: mp.Lock to prevent tearing during read/write.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        policy_version: torch.Tensor,
    ) -> None:
        self.shared_state = _build_shared_state_dict(policy, device)
        self.policy_version = policy_version
        self.lock = mp.Lock()

    def update(self, policy: torch.nn.Module) -> int:
        """Copy policy params → shared state and bump version. Returns new version."""
        with self.lock:
            _copy_model_to_state(policy, self.shared_state)
            self.policy_version += 1
        return int(self.policy_version.item())


class ParameterClient:
    """Used by InferenceServer to sync weights from shared state.

    Checks the policy_version counter; if stale, loads new weights
    under the same lock to avoid tearing.
    """

    def __init__(self, param_server: ParameterServer) -> None:
        self.shared_state = param_server.shared_state
        self.policy_version = param_server.policy_version
        self.lock = param_server.lock
        self._local_version: int = -1

    def ensure_updated(self, local_policy: torch.nn.Module) -> bool:
        """Load new weights into local_policy if version changed.

        Returns True if weights were updated, False if already current.
        """
        current = int(self.policy_version.item())
        if current == self._local_version:
            return False
        with self.lock:
            _load_state_into_model(self.shared_state, local_policy)
        self._local_version = current
        return True


# ---------------------------------------------------------------------------
# Internal helpers (inlined from the old utils/torch_state.py)
# ---------------------------------------------------------------------------

def _build_shared_state_dict(
    module: torch.nn.Module, device: torch.device
) -> "OrderedDict[str, torch.Tensor]":
    """Build a CPU shared-memory copy of the module's state dict.

    Always stored on CPU with share_memory_() so it works across spawned
    processes regardless of the training device. The copy/load helpers
    handle CPU↔GPU transfers automatically.
    """
    state = module.state_dict()
    shared: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state.items():
        t = tensor.detach().clone().cpu()
        t.requires_grad_(False)
        t.share_memory_()
        shared[name] = t
    return shared


def _copy_model_to_state(
    module: torch.nn.Module, target: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in target:
                dst = target[name]
                src = param.detach()
                if dst.device != src.device:
                    src = src.to(dst.device)
                dst.copy_(src)
        for name, buf in module.named_buffers():
            if name in target:
                dst = target[name]
                src = buf.detach()
                if dst.device != src.device:
                    src = src.to(dst.device)
                dst.copy_(src)


def _load_state_into_model(
    source: Mapping[str, torch.Tensor], module: torch.nn.Module
) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in source:
                src = source[name]
                if param.device != src.device:
                    src = src.to(param.device)
                param.copy_(src)
        for name, buf in module.named_buffers():
            if name in source:
                src = source[name]
                if buf.device != src.device:
                    src = src.to(buf.device)
                buf.copy_(src)
