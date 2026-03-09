# core/running_mean_std.py
"""Running mean/std trackers using Welford's online algorithm.

- RunningMeanStd: numpy-only, used by normalize_returns in the learner ingest thread.
- RunningMeanStdTorch: nn.Module with register_buffer, used by normalize_input.
  Buffers sync via ParameterServer (state_dict includes both params and buffers).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Tracks running mean and variance with parallel update (Welford's).

    Numpy-only — designed to run in the learner ingest thread (no GPU).
    """

    def __init__(self, epsilon: float = 1e-5, clip: float = 5.0):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon  # avoid div-by-zero on first call
        self.epsilon = epsilon
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        """Update running stats with a batch of scalar values."""
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Scale x to approximately zero-mean unit-variance."""
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip, self.clip,
        )

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Invert normalize: recover original scale."""
        return np.clip(x, -self.clip, self.clip) * np.sqrt(self.var + self.epsilon) + self.mean


class RunningMeanStdTorch(nn.Module):
    """Per-feature observation normalizer (PyTorch).

    Uses ``register_buffer`` so stats are included in ``state_dict()`` and
    automatically sync between Learner and InferenceServer via ParameterServer.

    Behaviour depends on ``self.training``:
      - ``train()`` mode (Learner): updates running stats AND normalizes.
      - ``eval()`` mode (InferenceServer): normalizes with frozen stats.
    """

    def __init__(self, shape: int, clip: float = 5.0, epsilon: float = 1e-5):
        super().__init__()
        self.clip = clip
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x)
        return self._normalize(x)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        """Welford's parallel update from a batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.running_mean
        total = self.count + batch_count
        new_mean = self.running_mean + delta * batch_count / total
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(m2 / total)
        self.count.copy_(total)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)).clamp(
            -self.clip, self.clip
        )
