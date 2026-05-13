"""Regression tests for atomic checkpoint writes.

The PPO/AMP/LSTM `save_checkpoint` methods used to call
`torch.save(ckpt, ckpt_path)` directly, which leaves a corrupt
final file if the process crashes mid-write. The fix writes to
`ckpt_path + ".tmp"` then `os.replace`s it onto the final path.

These tests pin that contract on the core PPO path; AMP and LSTM
share the same pattern (verified by grep) and rely on the same
torch / os.replace semantics.
"""

from __future__ import annotations

import os

import pytest
import torch

from beastrand.ppo.algorithm import PPOAlgorithm
from beastrand.ppo.policy import PPOPolicy


class _Cfg:
    obs_shape = (4,)
    act_shape = (2,)
    act_kind = "box"
    act_n = None

    class args:
        mlp_layers = [8, 8]
        use_lstm = False
        normalize_input = False
        normalize_returns = False
        learning_rate = 1e-3
        total_env_steps = 1024
        batch_size = 64
        lr_schedule = "constant"


def _make_algo():
    policy = PPOPolicy(_Cfg())
    opt = policy.build_optimizers(type("Ctx", (), {"args": _Cfg.args})())
    ctx = type("Ctx", (), {"args": _Cfg.args})()
    return PPOAlgorithm(ctx, policy, opt, torch.device("cpu")), policy


def test_save_checkpoint_leaves_no_tmp_file(tmp_path):
    algo, policy = _make_algo()
    ckpt_path = algo.save_checkpoint(str(tmp_path), policy, env_step=100)

    assert os.path.exists(ckpt_path), "final ckpt file should exist after save"
    assert not os.path.exists(ckpt_path + ".tmp"), "tmp file must be renamed away on success"


def test_save_checkpoint_round_trips(tmp_path):
    algo, policy = _make_algo()
    ckpt_path = algo.save_checkpoint(str(tmp_path), policy, env_step=200)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert loaded["env_step"] == 200
    assert "policy" in loaded and "opt" in loaded


def test_save_checkpoint_failure_leaves_final_path_untouched(tmp_path, monkeypatch):
    """If torch.save raises mid-write, the final ckpt path must not be
    created (only the .tmp may exist), preserving any prior good checkpoint.
    """
    algo, policy = _make_algo()
    # First, write a good checkpoint we can claim is "the prior one".
    good_path = algo.save_checkpoint(str(tmp_path), policy, env_step=10)
    good_bytes = open(good_path, "rb").read()

    # Now inject a failure into torch.save and try to overwrite the same step.
    real_save = torch.save

    def boom(obj, path, *a, **kw):
        # Write a few bytes to the .tmp so we can prove os.replace was skipped.
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("disk full (simulated)")

    monkeypatch.setattr("beastrand.ppo.algorithm.torch.save", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        algo.save_checkpoint(str(tmp_path), policy, env_step=10)

    # Restore for any cleanup
    monkeypatch.setattr("beastrand.ppo.algorithm.torch.save", real_save)

    # Prior good checkpoint must still load and match what we wrote.
    assert open(good_path, "rb").read() == good_bytes, (
        "failed save must not corrupt the previous final ckpt"
    )
