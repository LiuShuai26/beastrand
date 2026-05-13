"""Regression tests for beastrand.core.envs.make_env.

Covers:
- make_env does NOT mutate the process-global numpy RNG state. The factory
  previously called np.random.seed(seed) on every env construction, which
  created hidden coupling between env initialization and any code in the
  same worker that drew from np.random.
"""

import numpy as np
import pytest

from beastrand.core.envs.make_env import make_env


def test_make_env_does_not_pollute_global_numpy_rng():
    np.random.seed(12345)
    state_before = np.random.get_state()
    expected = np.random.random(4)

    np.random.set_state(state_before)
    env = make_env("CartPole-v1", seed=99)
    try:
        actual = np.random.random(4)
    finally:
        env.close()

    np.testing.assert_array_equal(
        actual, expected,
        err_msg="make_env must not advance the global np.random state",
    )


def test_make_env_seeds_action_space():
    """Action sampling is deterministic from the per-env seed even though
    the global RNG isn't touched."""
    env_a = make_env("CartPole-v1", seed=42)
    env_b = make_env("CartPole-v1", seed=42)
    try:
        seq_a = [env_a.action_space.sample() for _ in range(5)]
        seq_b = [env_b.action_space.sample() for _ in range(5)]
        assert seq_a == seq_b
    finally:
        env_a.close()
        env_b.close()
