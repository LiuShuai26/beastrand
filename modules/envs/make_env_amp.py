"""
AMP-specific environment factory.

Unlike the standard make_env, this does NOT apply NormalizeObservation or
NormalizeReward.  AMP requires raw (unnormalised) observations so the
algorithm can extract AMP state features that match reference motion data.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def make_env_amp(env_id: str, seed: int = 0, render_mode: str | None = None):
    """Create a Gymnasium environment for AMP training (no obs normalization).

    Args:
        env_id: Gym environment ID, e.g. "Humanoid-v5"
        seed:   RNG seed for env, action space, numpy
        render_mode: Optional render mode (e.g. "human", "rgb_array")

    Returns:
        env: Gymnasium environment instance
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    # No NormalizeObservation — AMP needs raw obs for feature extraction.
    # No NormalizeReward — task rewards used directly alongside style rewards.

    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    return env
