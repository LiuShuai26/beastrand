"""Regression tests for PettingZooParallelWrapper.

Covers the multi-agent side-channel protocol and the optional
``mirror_h_for_opponents`` flag added to support symmetric two-player Atari
envs (e.g. pong_v3) where both agents receive an identical pixel buffer.
A shared self-play policy cannot disambiguate which paddle it controls from
identical input; horizontally mirroring agent 1+ obs lets one policy learn a
single "my paddle on the left" perspective.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from projects.tennis.pettingzoo_wrapper import PettingZooParallelWrapper


class FakeParallelEnv:
    """Minimal PettingZoo ParallelEnv stand-in with deterministic per-step obs.

    Each step returns the same (C, H, W) frame for both agents, with content
    derived from the step counter so we can verify identity and mirroring.
    """

    possible_agents = ["first_0", "second_0"]

    def __init__(self, shape=(4, 84, 84)):
        self._shape = shape
        self._t = 0
        self._obs_space = gym.spaces.Box(0, 255, shape, dtype=np.uint8)
        self._act_space = gym.spaces.Discrete(6)

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def _frame(self) -> np.ndarray:
        # Asymmetric pattern along the W axis so a horizontal flip is detectable.
        frame = np.zeros(self._shape, dtype=np.uint8)
        frame[:, :, 0] = 200  # bright stripe on the LEFT edge of W
        frame[:, :, -1] = 50  # dim stripe on the RIGHT edge of W
        frame[:, :, self._t % self._shape[-1]] = 255  # moving marker
        return frame

    def reset(self, seed=None, options=None):
        self._t = 0
        f = self._frame()
        return {a: f.copy() for a in self.possible_agents}, {a: {} for a in self.possible_agents}

    def step(self, actions_dict):
        self._t += 1
        f = self._frame()
        return (
            {a: f.copy() for a in self.possible_agents},
            {a: 0.0 for a in self.possible_agents},
            {a: False for a in self.possible_agents},
            {a: False for a in self.possible_agents},
            {a: {} for a in self.possible_agents},
        )

    def close(self):
        pass


def test_default_no_mirror_back_compat():
    """Default behavior unchanged: both agents see identical obs."""
    env = PettingZooParallelWrapper(FakeParallelEnv())
    env.reset(seed=0)
    assert np.array_equal(env.agent_obs[0], env.agent_obs[1])
    env.agent_actions[1] = 0
    env.step(0)
    assert np.array_equal(env.agent_obs[0], env.agent_obs[1])


def test_mirror_h_for_opponents_flips_agent_1_obs():
    """With mirror flag, agent 1 obs is the W-axis flip of agent 0."""
    env = PettingZooParallelWrapper(FakeParallelEnv(), mirror_h_for_opponents=True)
    env.reset(seed=0)
    o0, o1 = env.agent_obs[0], env.agent_obs[1]
    assert o0.shape == o1.shape, "mirror must preserve shape"
    assert o0.dtype == o1.dtype, "mirror must preserve dtype"
    assert not np.array_equal(o0, o1), "mirrored opponent must differ from agent 0"
    assert np.array_equal(o0, o1[..., ::-1]), "agent 1 obs must be the H-flip of agent 0"
    assert o1.flags.c_contiguous, "agent 1 obs must be contiguous for downstream torch.from_numpy"


def test_mirror_h_for_opponents_holds_across_steps():
    """Flip invariant survives reset → step → step transitions."""
    env = PettingZooParallelWrapper(FakeParallelEnv(), mirror_h_for_opponents=True)
    env.reset(seed=0)
    for _ in range(5):
        env.agent_actions[1] = 0
        env.step(0)
        o0, o1 = env.agent_obs[0], env.agent_obs[1]
        assert np.array_equal(o0, o1[..., ::-1])


def test_mirror_h_does_not_affect_returned_agent_0_obs():
    """Only side-channel agent_obs is mirrored; the Gym-API return (agent 0) is untouched."""
    env = PettingZooParallelWrapper(FakeParallelEnv(), mirror_h_for_opponents=True)
    obs, _ = env.reset(seed=0)
    assert np.array_equal(obs, env.agent_obs[0])
    env.agent_actions[1] = 0
    next_obs, _, _, _, _ = env.step(0)
    assert np.array_equal(next_obs, env.agent_obs[0])
