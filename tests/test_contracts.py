"""Contract tests: verify DataRecord ↔ Policy ↔ Algorithm consistency.

Two test classes:
  1. TestModuleContracts — fast, no env needed. Validates alloc_specs → build_batch
     key flow and policy interface for each project config.
  2. TestProjectSmoke — parametrized 1-iteration smoke test per project.
     Marked @pytest.mark.slow; skips if env deps are missing.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import fields

import numpy as np
import pytest
import torch

from beastrand.core.contract import validate_contracts
from beastrand.utils.import_utils import get_object_from_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(args):
    """Build a minimal Shared-like context from an Args config."""
    from beastrand.nodes.manager import probe_env, Shared

    ctx = Shared(args)
    _mep = getattr(args, "make_env_path", None)
    specs = probe_env(args.env_id, seed=0, make_env_path=_mep, args=args)
    obs_spec, act_spec = specs["obs"], specs["act"]

    ctx.obs_shape = tuple(obs_spec["shape"])
    ctx.obs_dtype = obs_spec["dtype"]
    ctx.act_kind = act_spec["kind"]
    ctx.act_shape = tuple(act_spec["shape"])
    ctx.act_dtype = act_spec["dtype"]
    ctx.act_n = act_spec["n"]
    ctx.act_low = act_spec["low"]
    ctx.act_high = act_spec["high"]
    return ctx


def _make_test_args(args_cls, overrides=None):
    """Instantiate an Args config with minimal values for testing."""
    defaults = {
        "total_env_steps": 128,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "rollout": 32,
        "batch_size": 32,
        "minibatch_size": 32,
        "replay_capacity": 32,
        "learning_starts": 32,
        "train_epochs": 1,
        "inference_device": "cpu",
        "learner_device": "cpu",
        "normalize_input": False,
        "normalize_returns": False,
        "checkpoint_interval": 0,
        "run_name": "test_contract",
    }
    if overrides:
        defaults.update(overrides)

    # Only set fields that actually exist on this config class
    field_names = {f.name for f in fields(args_cls) if f.init}
    kwargs = {k: v for k, v in defaults.items() if k in field_names}
    return args_cls(**kwargs)


# ---------------------------------------------------------------------------
# Project definitions: (name, args_cls_path, extra_overrides, env_dep)
# ---------------------------------------------------------------------------
_PROJECTS = [
    ("ppo", "beastrand.ppo.config.Args", {"env_id": "CartPole-v1"}, None),
    ("mujoco", "projects.mujoco.config.Args", {"env_id": "HalfCheetah-v4"}, "mujoco"),
    ("atari", "projects.atari.config.Args", {"env_id": "PongNoFrameskip-v4"}, "ale_py"),
    ("ppo_lstm", "projects.ppo_lstm.config.Args", {"env_id": "CartPole-v1"}, None),
]


def _get_project_params():
    """Build pytest params, skipping projects whose env deps are missing."""
    params = []
    for name, args_path, overrides, dep in _PROJECTS:
        marks = []
        if dep is not None:
            marks.append(pytest.mark.skipif(
                not _has_module(dep),
                reason=f"{dep} not installed",
            ))
        params.append(pytest.param(args_path, overrides, id=name, marks=marks))
    return params


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ===========================================================================
# 1. Module Contract Tests (fast, no training)
# ===========================================================================

class TestModuleContracts:
    """Validate alloc_specs → build_batch → policy.act key contracts."""

    @pytest.mark.parametrize("args_path,overrides", _get_project_params())
    def test_contract_validation(self, args_path, overrides):
        """validate_contracts() should pass without errors for each project."""
        args_cls = get_object_from_path(args_path)
        args = _make_test_args(args_cls, overrides)
        ctx = _make_ctx(args)

        record_cls = get_object_from_path(args.data_record_path)
        policy_cls = get_object_from_path(args.policy_path)
        algorithm_cls = get_object_from_path(args.algorithm_path)

        # Should not raise
        validate_contracts(ctx, record_cls, policy_cls, algorithm_cls)

    @pytest.mark.parametrize("args_path,overrides", _get_project_params())
    def test_build_batch_keys_are_valid(self, args_path, overrides):
        """build_batch output keys should all have non-empty numpy arrays."""
        args_cls = get_object_from_path(args_path)
        args = _make_test_args(args_cls, overrides)
        ctx = _make_ctx(args)

        record_cls = get_object_from_path(args.data_record_path)
        T = args.rollout
        specs = record_cls.alloc_specs(ctx, T, ctx.obs_shape, ctx.act_shape)

        fake_view = {}
        for field, (shape, dtype_str) in specs.items():
            fake_view[field] = np.zeros(shape, dtype=np.dtype(dtype_str))

        batch = record_cls.build_batch(ctx, fake_view)

        for key, arr in batch.items():
            assert isinstance(arr, np.ndarray), f"batch['{key}'] is not ndarray"
            assert arr.size > 0, f"batch['{key}'] is empty"

    @pytest.mark.parametrize("args_path,overrides", _get_project_params())
    def test_policy_act_returns_action(self, args_path, overrides):
        """policy.act() must return 'action' key with correct batch dim."""
        args_cls = get_object_from_path(args_path)
        args = _make_test_args(args_cls, overrides)
        ctx = _make_ctx(args)

        policy_cls = get_object_from_path(args.policy_path)
        policy = policy_cls(ctx)
        policy.eval()

        B = 4
        with torch.no_grad():
            out = policy.act({"obs": torch.zeros(B, *ctx.obs_shape)}, deterministic=True)

        assert "action" in out, f"act() missing 'action', got: {set(out.keys())}"
        assert out["action"].shape[0] == B

    @pytest.mark.parametrize("args_path,overrides", _get_project_params())
    def test_policy_evaluate_actions_returns_required(self, args_path, overrides):
        """policy.evaluate_actions() must return logp, entropy, value."""
        args_cls = get_object_from_path(args_path)
        args = _make_test_args(args_cls, overrides)
        ctx = _make_ctx(args)

        policy_cls = get_object_from_path(args.policy_path)
        policy = policy_cls(ctx)
        policy.train()

        B = 4
        fake_obs = torch.zeros(B, *ctx.obs_shape)
        with torch.no_grad():
            act_out = policy.act({"obs": fake_obs})

        inputs = {"obs": fake_obs, "action": act_out["action"]}
        eval_out = policy.evaluate_actions(inputs)

        for key in ("logp", "entropy", "value"):
            assert key in eval_out, (
                f"evaluate_actions() missing '{key}', got: {set(eval_out.keys())}"
            )


# ===========================================================================
# 2. Project Smoke Tests (1 iteration, marked slow)
# ===========================================================================

def _run_smoke(args_path, overrides, result_dict):
    """Run 1 PPO iteration in a subprocess."""
    from beastrand.core.common import set_start_method
    set_start_method()

    args_cls = get_object_from_path(args_path)
    args = _make_test_args(args_cls, overrides)
    if hasattr(args, "validate"):
        args.validate()

    from beastrand.nodes.manager import Manager
    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()
    result_dict["steps"] = mgr.ctx.global_step.value


# Only smoke-test projects that don't need special env deps
_SMOKE_PROJECTS = [
    pytest.param(
        "beastrand.ppo.config.Args",
        {"env_id": "CartPole-v1"},
        id="ppo_cartpole",
    ),
    pytest.param(
        "projects.ppo_lstm.config.Args",
        {"env_id": "Pendulum-v1", "lstm_hidden_size": 32, "recurrence": 32},
        id="ppo_lstm_pendulum",
    ),
]


class TestProjectSmoke:
    """Run a minimal 1-iteration training for each project."""

    @pytest.mark.slow
    @pytest.mark.parametrize("args_path,overrides", _SMOKE_PROJECTS)
    def test_smoke_one_iteration(self, args_path, overrides):
        """Project completes 128 env steps without crashing."""
        spawn_ctx = mp.get_context("spawn")
        result_dict = spawn_ctx.Manager().dict()

        p = spawn_ctx.Process(
            target=_run_smoke,
            args=(args_path, overrides, result_dict),
        )
        p.start()
        p.join(timeout=60)

        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            pytest.fail("Smoke test timed out after 60s")

        assert p.exitcode == 0, f"Training process exited with code {p.exitcode}"
        assert result_dict.get("steps", 0) >= 128
