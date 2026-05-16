# projects/atari/make_env.py
"""Atari environment factory with standard preprocessing wrappers (SF-matched)."""

from __future__ import annotations

import ale_py  # noqa: F401  — registers ALE envs in Gymnasium
import gymnasium as gym
import numpy as np


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1} by its sign (SF-style)."""
    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


def make_env(env_id: str, seed: int = 0, args=None):
    """Create an Atari env with SF-matched preprocessing.

    Wrapper stack:
      RecordEpisodeStatistics → AtariPreprocessing(noop=30, skip=4,
      84x84, grayscale, terminal_on_life_loss) → ClipRewardEnv →
      FrameStack(4)

    Final obs shape: (4, 84, 84) uint8.
    """
    env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,       # keep uint8, scale via obs_normalizer or policy
    )
    env = ClipRewardEnv(env)

    frame_stack = getattr(args, "frame_stack", 4) if args else 4
    env = gym.wrappers.FrameStackObservation(env, frame_stack)

    return env
