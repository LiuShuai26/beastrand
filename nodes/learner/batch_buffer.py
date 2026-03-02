# batch_buffer.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np


class BatchBuffer:
    """
    Step-major circular buffer that follows server schema directly.
    Fields/shape/dtype all come from meta["schema"].
    """

    def __init__(self, meta: Dict[str, Any], batch_size: int, buffer_size: int):
        self.schema = dict(meta["schema"])
        self.T = int(meta["T"])
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.write_pos = 0
        self.valid_steps = 0

        # allocate arrays for every schema field
        self.arrays: Dict[str, np.ndarray] = {}
        for field, spec in self.schema.items():
            shape = (self.buffer_size, *spec["shape"][1:])
            dtype = np.dtype(spec["dtype"])

            self.arrays[field] = np.zeros(shape, dtype=dtype)

    def _span(self, n: int):
        s = self.write_pos
        if s + n <= self.buffer_size:
            return [(slice(s, s + n), 0, n)]
        first = self.buffer_size - s
        return [(slice(s, self.buffer_size), 0, first), (slice(0, n - first), first, n - first)]

    def append_slot(self, slot_arrays: Dict[str, np.ndarray]):
        spans = self._span(self.T)
        for sl, off, n in spans:
            for f, arr in self.arrays.items():
                # pick time-major field (T or T+1); here just slice [off:off+n]
                self.arrays[f][sl] = slot_arrays[f][off:off+n]
        self.write_pos = (self.write_pos + self.T) % self.buffer_size
        self.valid_steps = min(self.valid_steps + self.T, self.buffer_size)

    def get_batch(self) -> Dict[str, np.ndarray]:
        """Return the entire buffer as a copy. Raises if not yet full."""
        if self.valid_steps < self.buffer_size:
            raise RuntimeError(
                f"BatchBuffer not full yet: valid={self.valid_steps}, capacity={self.buffer_size}"
            )

        return {f: arr.copy() for f, arr in self.arrays.items()}
