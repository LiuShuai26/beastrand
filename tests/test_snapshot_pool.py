"""Regression tests for SnapshotPool prune/load race handling.

The learner's _prune can delete a snapshot file between an opponent
inference server's sample_random() and load_into(). The fix: load_into
returns False on FileNotFoundError, and sample_and_load retries with a
fresh sample.
"""

import os

import pytest
import torch
import torch.nn as nn

from beastrand.utils.snapshot_pool import SnapshotPool


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_load_into_returns_true_on_success(tmp_path):
    pool = SnapshotPool(str(tmp_path))
    policy = _TinyPolicy()
    path = pool.save(policy, version=0)
    target = _TinyPolicy()
    # Perturb target weights so the load is observable.
    with torch.no_grad():
        target.fc.weight.fill_(99.0)
    assert pool.load_into(path, target) is True
    assert torch.allclose(target.fc.weight, policy.fc.weight)


def test_load_into_returns_false_when_pruned(tmp_path):
    pool = SnapshotPool(str(tmp_path))
    policy = _TinyPolicy()
    path = pool.save(policy, version=0)
    # Simulate race: prune deletes the file before load_into reads it.
    os.remove(path)
    assert pool.load_into(path, _TinyPolicy()) is False


def test_sample_and_load_retries_through_race(tmp_path, monkeypatch):
    pool = SnapshotPool(str(tmp_path))
    policy = _TinyPolicy()
    path = pool.save(policy, version=0)
    # Inject one pruned sample, then a real one. sample_and_load should
    # absorb the prune race and succeed on retry.
    samples = iter([path + ".gone", path])
    monkeypatch.setattr(pool, "sample_random", lambda: next(samples))
    target = _TinyPolicy()
    assert pool.sample_and_load(target) is True
    assert torch.allclose(target.fc.weight, policy.fc.weight)


def test_sample_and_load_empty_pool(tmp_path):
    pool = SnapshotPool(str(tmp_path))
    assert pool.sample_and_load(_TinyPolicy()) is False


def test_sample_and_load_gives_up_after_max_retries(tmp_path, monkeypatch):
    """Every sample races with prune. Bounded retry returns False (not infinite loop)."""
    pool = SnapshotPool(str(tmp_path))
    pool.save(_TinyPolicy(), version=0)
    monkeypatch.setattr(pool, "sample_random", lambda: str(tmp_path / "always_pruned"))
    target = _TinyPolicy()
    assert pool.sample_and_load(target, max_retries=2) is False
