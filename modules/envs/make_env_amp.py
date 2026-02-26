"""
AMP-specific environment factory.

Supports two environment backends:
  1. Beast .so environments (e.g. HumanoidEnv) — loaded via beastlab.env_loader
  2. Standard Gymnasium environments (e.g. Humanoid-v5) — loaded via gym.make

The factory tries Beast first, falling back to gym.make if the .so is not found
or beastlab is not installed.

Unlike the standard make_env, this does NOT apply NormalizeObservation or
NormalizeReward.  AMP requires raw (unnormalised) observations so the
algorithm can extract AMP state features that match reference motion data.
"""

from __future__ import annotations

import logging
import gymnasium as gym
import numpy as np


def make_env_amp(env_id: str, seed: int = 0, render_mode: str | None = None):
    """Create an environment for AMP training (no obs normalization).

    Tries to load a Beast .so environment first. If beastlab is not installed
    or the .so module is not found, falls back to ``gym.make(env_id)``.

    Args:
        env_id: Beast module name (e.g. "HumanoidEnv") or Gym env ID (e.g. "Humanoid-v5")
        seed:   RNG seed for env, action space, numpy
        render_mode: Optional render mode (e.g. "human", "rgb_array")

    Returns:
        env: Gymnasium-compatible environment instance
    """
    env = _try_beast_env(env_id)

    if env is None:
        # Standard Gymnasium env
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


def _try_beast_env(env_id: str):
    """Attempt to load a Beast .so environment. Returns None on failure."""
    try:
        from beastlab.env_loader import make_beast_gym
        env = make_beast_gym(env_id)
        logging.info("Loaded Beast .so env: %s", env_id)
        return env
    except ImportError:
        return None
    except FileNotFoundError:
        return None
