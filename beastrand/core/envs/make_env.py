# core/envs/make_env.py
"""Minimal gymnasium environment factory.

Normalization is handled at the framework level, not per-worker:
  - Observation normalization: policy-level RunningMeanStdTorch (--normalize-input)
  - Value target normalization: learner-level RunningMeanStd (--normalize-returns)
"""

from __future__ import annotations
import gymnasium as gym


def make_env(env_id: str, seed: int = 0, render_mode: str | None = None,
             *, args=None):
    """
    Create a Gymnasium environment with reproducible seeding.

    Args:
        env_id: Gym environment ID, e.g. "CartPole-v1"
        seed: RNG seed for the env and its action space.
        render_mode: Optional render mode (e.g. "human", "rgb_array")

    Returns:
        env: Gymnasium environment instance
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env
