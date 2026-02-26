# modules/dataset/base_record.py
from __future__ import annotations
from typing import Dict, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod

class DataRecordBase(ABC):
    """
    Contract:
      - alloc_specs(ctx, T, obs_shape, act_shape) -> {field: ((...), dtype_str)}
      - build_batch(ctx, view) -> Dict[str, np.ndarray] (algo-specific keys)
    """

    @staticmethod
    @abstractmethod
    def alloc_specs(ctx, T: int, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...]) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_batch(ctx: Any, view: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError
