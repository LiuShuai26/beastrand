"""Tests for projects.pong.dual_env.DualEnvWrapper.

Covers mode switching, the agent_obs[1]=None contract in bot mode, and the
H-flip applied to the atari obs (which keeps the policy on a single
"my paddle on the left" perspective shared with mirrored pong_v3).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from projects.pong.dual_env import DualEnvWrapper


class FakePettingZooEnv(gym.Env):
    """Stand-in for PettingZooParallelWrapper with the multi-agent contract."""

    def __init__(self, shape=(4, 84, 84)):
        super().__init__()
        self._shape = shape
        self.observation_space = gym.spaces.Box(0, 255, shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)
        self.num_agents = 2
        self.agent_obs = [None, None]
        self.agent_actions = [None, None]
        self._t = 0

    def _frames(self):
        f0 = np.zeros(self._shape, dtype=np.uint8)
        f0[:, :, 0] = 200   # bright stripe LEFT
        f0[:, :, -1] = 10   # dim RIGHT
        # agent 1 is the H-flip — matches mirror=True semantics
        f1 = np.ascontiguousarray(f0[..., ::-1])
        return f0, f1

    def reset(self, *, seed=None, options=None):
        self._t = 0
        f0, f1 = self._frames()
        self.agent_obs = [f0, f1]
        return f0, {}

    def step(self, action):
        self._t += 1
        f0, f1 = self._frames()
        self.agent_obs = [f0, f1]
        return f0, 0.0, False, False, {}

    def close(self):
        pass


class FakeAtariEnv(gym.Env):
    """Stand-in for single-agent PongNoFrameskip-v4 with Atari preprocessing."""

    def __init__(self, shape=(4, 84, 84)):
        super().__init__()
        self._shape = shape
        self.observation_space = gym.spaces.Box(0, 255, shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)

    def _frame(self):
        # Distinct pattern from the pettingzoo fake so we can tell envs apart.
        f = np.zeros(self._shape, dtype=np.uint8)
        f[:, :, 5] = 123
        f[:, 10, :] = 77
        return f

    def reset(self, *, seed=None, options=None):
        return self._frame(), {}

    def step(self, action):
        return self._frame(), 1.0, False, False, {}

    def close(self):
        pass


def _make_wrapper():
    return DualEnvWrapper(FakePettingZooEnv(), FakeAtariEnv())


def test_default_mode_is_snapshot_and_uses_pong_v3():
    env = _make_wrapper()
    obs, _ = env.reset(seed=0)
    # Default routes through pong_v3 → agent_obs[1] populated (H-flipped frame)
    assert env.agent_obs[0] is not None
    assert env.agent_obs[1] is not None
    assert env.agent_obs[0].shape == (4, 84, 84)
    # agent 1 obs is the mirror of agent 0 (per fake pettingzoo env)
    assert np.array_equal(env.agent_obs[0], env.agent_obs[1][..., ::-1])


def test_bot_mode_sets_agent_obs_1_to_none():
    env = _make_wrapper()
    env.set_opponent_mode("bot")
    obs, _ = env.reset(seed=0)
    assert env.agent_obs[0] is not None
    assert env.agent_obs[1] is None
    # H-flip applied to atari obs — should NOT match raw atari frame.
    raw_atari = FakeAtariEnv()._frame()
    assert not np.array_equal(env.agent_obs[0], raw_atari)
    assert np.array_equal(env.agent_obs[0], raw_atari[..., ::-1])


def test_mode_switch_takes_effect_on_reset():
    env = _make_wrapper()
    env.reset(seed=0)
    pong_frame = env.agent_obs[0].copy()

    env.set_opponent_mode("bot")
    env.reset(seed=0)
    bot_frame = env.agent_obs[0]

    # The active env actually changed — frames differ in content.
    assert not np.array_equal(pong_frame, bot_frame)
    assert env.agent_obs[1] is None

    # Switching back restores pong path.
    env.set_opponent_mode("snapshot")
    env.reset(seed=0)
    assert env.agent_obs[1] is not None


def test_bot_step_ignores_agent_actions_1():
    env = _make_wrapper()
    env.set_opponent_mode("bot")
    env.reset(seed=0)
    # Worker is supposed to NOT write agent_actions[1] in bot mode; verify
    # leaving it None and stepping doesn't fault.
    env.agent_actions[1] = None
    obs, r, term, trunc, info = env.step(0)
    assert env.agent_obs[1] is None
    assert obs.shape == (4, 84, 84)


def test_invalid_mode_rejected():
    env = _make_wrapper()
    with pytest.raises(ValueError):
        env.set_opponent_mode("garbage")
