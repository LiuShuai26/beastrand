"""Regression tests for PPO config validation.

Specifically guards the learning_starts vs replay_capacity invariant: the
learner's `valid_steps >= learning_starts` trigger requires equality
because BatchBuffer.valid_steps caps at replay_capacity. The prior check
only rejected `learning_starts < replay_capacity`, allowing the
`learning_starts > replay_capacity` configuration that hangs forever.

Covers all standalone Args dataclasses: each project (AMP / tennis /
self-play) defines its own and re-implemented the same buggy check, so
each needs explicit coverage.
"""

import pytest


# Per-config kwargs (defaults aligned with each Args's own field defaults
# so equal learning_starts/replay_capacity is the baseline). Each tuple is
# (import_path, base_kwargs).
_CONFIGS = []


def _register_base():
    from beastrand.ppo.config import Args
    _CONFIGS.append((
        "base",
        Args,
        dict(replay_capacity=1024, learning_starts=1024, batch_size=1024,
             minibatch_size=256, rollout=64),
    ))


def _register_amp():
    from projects.ppo_amp.config import Args as AMPArgs
    _CONFIGS.append((
        "amp",
        AMPArgs,
        dict(replay_capacity=8192, learning_starts=8192, batch_size=8192,
             minibatch_size=512, rollout=32,
             # AMP-specific required fields; values not exercised here.
             keyframe_file="dummy.json", amp_obs_slices=[(0, 1)]),
    ))


def _register_tennis():
    from projects.tennis.config import Args as TennisArgs
    _CONFIGS.append((
        "tennis",
        TennisArgs,
        dict(replay_capacity=1024, learning_starts=1024, batch_size=1024,
             minibatch_size=256, rollout=64),
    ))


def _register_selfplay():
    from projects.ppo_selfplay.config import Args as SelfPlayArgs
    _CONFIGS.append((
        "selfplay",
        SelfPlayArgs,
        dict(replay_capacity=1024, learning_starts=1024, batch_size=1024,
             minibatch_size=256, rollout=64),
    ))


# Eager registration so parametrize can see the list.
_register_base()
_register_amp()
_register_tennis()
_register_selfplay()


@pytest.mark.parametrize("name,cls,base", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_learning_starts_equal_replay_capacity_ok(name, cls, base):
    cls(**base)  # no exception


@pytest.mark.parametrize("name,cls,base", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_learning_starts_greater_than_replay_capacity_raises(name, cls, base):
    """The codex-flagged hang case: learner trigger never fires."""
    kw = dict(base, learning_starts=base["replay_capacity"] * 2)
    with pytest.raises(ValueError, match="learning_starts must equal replay_capacity"):
        cls(**kw)


@pytest.mark.parametrize("name,cls,base", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_learning_starts_less_than_replay_capacity_raises(name, cls, base):
    """The other broken direction: trigger fires before buffer full,
    get_batch then raises in the learner main thread."""
    kw = dict(base, learning_starts=base["replay_capacity"] // 2)
    with pytest.raises(ValueError, match="learning_starts must equal replay_capacity"):
        cls(**kw)


def test_manager_guard_fallback():
    """Independent of project config validation: Manager.launch() applies
    the same invariant. Reading the source confirms the check is wired in
    — a regression integration test would need a real env/process tree,
    which is out of scope; we settle for source-level proof so the guard
    can't be silently deleted."""
    import inspect
    from beastrand.nodes import manager

    src = inspect.getsource(manager.Manager.launch)
    assert "learning_starts must equal replay_capacity" in src
