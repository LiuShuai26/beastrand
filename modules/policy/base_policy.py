from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

class BasePolicy(ABC, nn.Module):
    """
    Unified policy interface for inference server.
    All policies must implement `act`. Other methods are capability-based.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.obs_dim: int = int(np.prod(np.array(cfg.obs_shape)))
        self.act_dim: int = int(np.prod(np.array(cfg.act_shape)))

    # ---- required ----
    @abstractmethod
    def act(self, inputs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: dictionary that must contain key ``"obs"`` mapped to a tensor of
                shape ``[B, obs_dim]``. Additional keys (e.g. ``"rnn_state"`` or
                ``"mask"``) are policy-specific and optional.
            deterministic: if True, return the mean action instead of sampling.

        Returns:
            A payload dict with any of:
              - "action": Tensor [B, act_dim]  (required)
              - "logp"  : Tensor [B]           (optional)
              - "value" : Tensor [B]           (optional; only if supports_value())
              - policy-specific extras (e.g. "rnn_state")
        """
        ...

    # ---- optional / capability-based ----
    def supports_value(self) -> bool:
        """Override to True for critics that expose V(s) (e.g., PPO)."""
        return False

    def value(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Only valid if supports_value() is True."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support value()")

    def evaluate_actions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        PPO-style utility on fixed actions.
        Non-PPO policies may ignore or raise NotImplementedError.

        Args:
            inputs: dictionary containing at least ``"obs"`` and ``"action"``. Extra
                keys are policy specific (e.g. ``"rnn_state"``).

        Returns:
            Dictionary with entries such as ``"logp"``, ``"entropy"`` and
            ``"value"`` depending on the policy's capabilities.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support evaluate_actions()")
