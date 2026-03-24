"""Snapshot pool for self-play opponent sampling.

Manages a directory of historical policy checkpoints on disk.
Learner saves snapshots periodically; opponent InferenceServer loads them.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
from typing import List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class SnapshotPool:
    """Manages policy snapshots for self-play opponent sampling.

    Snapshots are saved as ``state_dict()`` only (no optimizer state).
    Uses atomic write (write to temp file, then rename) for crash safety.
    FIFO pruning keeps the pool bounded.
    """

    PREFIX = "snap_v"
    SUFFIX = ".pt"

    def __init__(self, snapshot_dir: str, max_snapshots: int = 20) -> None:
        self.snapshot_dir = snapshot_dir
        self.max_snapshots = max_snapshots
        os.makedirs(snapshot_dir, exist_ok=True)

    def save(self, policy: nn.Module, version: int) -> str:
        """Save policy state_dict as a snapshot. Returns the saved path."""
        filename = f"{self.PREFIX}{version:08d}{self.SUFFIX}"
        final_path = os.path.join(self.snapshot_dir, filename)

        # Atomic write: temp file + rename in same directory
        fd, tmp_path = tempfile.mkstemp(
            dir=self.snapshot_dir, suffix=".tmp"
        )
        try:
            os.close(fd)
            torch.save(policy.state_dict(), tmp_path)
            os.replace(tmp_path, final_path)
        except BaseException:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        self._prune()
        log.info("Snapshot saved: %s", filename)
        return final_path

    def list_snapshots(self) -> List[str]:
        """Return sorted list of snapshot file paths (oldest first)."""
        if not os.path.isdir(self.snapshot_dir):
            return []
        files = [
            os.path.join(self.snapshot_dir, f)
            for f in sorted(os.listdir(self.snapshot_dir))
            if f.startswith(self.PREFIX) and f.endswith(self.SUFFIX)
        ]
        return files

    def sample_random(self) -> Optional[str]:
        """Sample a random snapshot path. Returns None if pool is empty."""
        snaps = self.list_snapshots()
        if not snaps:
            return None
        return random.choice(snaps)

    def load_into(self, path: str, policy: nn.Module) -> None:
        """Load snapshot weights into a policy model."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state)
        log.info("Snapshot loaded: %s", os.path.basename(path))

    def _prune(self) -> None:
        """Remove oldest snapshots exceeding max_snapshots."""
        if self.max_snapshots <= 0:
            return
        snaps = self.list_snapshots()
        while len(snaps) > self.max_snapshots:
            oldest = snaps.pop(0)
            os.remove(oldest)
            log.debug("Pruned snapshot: %s", os.path.basename(oldest))
