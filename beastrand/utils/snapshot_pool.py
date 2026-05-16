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

    def load_into(self, path: str, policy: nn.Module) -> bool:
        """Load snapshot weights into a policy model.

        Returns True on success, False if the snapshot file was pruned
        between sampling and loading (race against ``_prune``). Callers
        should retry with a fresh sample on False.
        """
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            log.debug(
                "Snapshot %s pruned before load — caller should resample",
                os.path.basename(path),
            )
            return False
        policy.load_state_dict(state)
        log.info("Snapshot loaded: %s", os.path.basename(path))
        return True

    def sample_and_load(
        self, policy: nn.Module, max_retries: int = 3
    ) -> bool:
        """Sample a random snapshot and load it, retrying on prune races.

        Returns True on success, False if no snapshot could be loaded after
        ``max_retries`` attempts (typically: pool is empty or every sample
        races with prune — both rare).
        """
        for _ in range(max_retries):
            path = self.sample_random()
            if path is None:
                return False
            if self.load_into(path, policy):
                return True
        return False

    def _prune(self) -> None:
        """Remove oldest snapshots exceeding max_snapshots."""
        if self.max_snapshots <= 0:
            return
        snaps = self.list_snapshots()
        while len(snaps) > self.max_snapshots:
            oldest = snaps.pop(0)
            os.remove(oldest)
            log.debug("Pruned snapshot: %s", os.path.basename(oldest))
