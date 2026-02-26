"""
AMP-specific environment factory.

Supports two environment backends:
  1. Beast .so environments (e.g. HumanoidEnv) — loaded via importlib + BeastGymWrapper
  2. Standard Gymnasium environments (e.g. Humanoid-v5) — loaded via gym.make

The factory tries Beast first, falling back to gym.make if the .so is not found.

Unlike the standard make_env, this does NOT apply NormalizeObservation or
NormalizeReward.  AMP requires raw (unnormalised) observations so the
algorithm can extract AMP state features that match reference motion data.

This module is fully standalone — no external beastlab dependency required.
BeastGymWrapper is inlined below to wrap raw Beast .so environments with
the standard Gymnasium interface.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BeastGymWrapper(gym.Env):
    """Gymnasium wrapper for raw Beast .so environments.

    Beast .so environments expose a C++ pybind11 interface:
      - observation_size() -> int
      - action_space() -> dict with "continuous_size"
      - reset(seed=None) -> (obs, info)
      - step(action) -> (obs, rewards, terminated, truncated, info)

    This wrapper adapts that interface to standard Gymnasium (Box spaces,
    scalar reward, bool done flags, np.float32 arrays).
    """

    def __init__(self, raw_env):
        super().__init__()
        self._env = raw_env
        self._env.reset()

        obs_size = self._env.observation_size()
        action_info = self._env.action_space()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32,
        )
        cont_size = action_info["continuous_size"]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cont_size,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, info = self._env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, rewards, terminated, truncated, info = self._env.step(action)
        return (
            np.asarray(obs, dtype=np.float32),
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )


def make_env_amp(env_id: str, seed: int = 0, render_mode: str | None = None,
                 *, args=None):
    """Create an environment for AMP training (no obs normalization).

    Tries to load a Beast .so environment first. If the .so module is not
    found, falls back to ``gym.make(env_id)``.

    Args:
        env_id: Beast module name (e.g. "HumanoidEnv") or Gym env ID (e.g. "Humanoid-v5")
        seed:   RNG seed for env, action space, numpy
        render_mode: Optional render mode (e.g. "human", "rgb_array")
        args:   Optional config object with Beast-specific fields
                (reward_mode, keyframe_file, target_vx)

    Returns:
        env: Gymnasium-compatible environment instance
    """
    env = _try_beast_env(env_id, args=args)

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


_BEAST_CONFIGURED = False


def _configure_beast_statics(env_module, args) -> None:
    """Set Beast .so module-level statics BEFORE env construction.

    These calls configure the C++ side (reward mode, keyframe file, target
    velocity) and must run once per process before any env instance is created.
    Gracefully skips if the .so does not expose the expected static functions.
    """
    global _BEAST_CONFIGURED
    if _BEAST_CONFIGURED:
        return

    MODE_MAP = {"idle": 0, "walk": 1, "punch": 2}
    reward_mode = getattr(args, "reward_mode", "idle")

    if hasattr(env_module, "set_reward_mode"):
        assert reward_mode in MODE_MAP, f"unknown reward mode: {reward_mode}"
        env_module.set_reward_mode(MODE_MAP[reward_mode])

    keyframe_file = getattr(args, "keyframe_file", "")
    if keyframe_file and hasattr(env_module, "set_keyframe_file"):
        env_module.set_keyframe_file(str(Path(keyframe_file).resolve()))

    if reward_mode == "walk" and hasattr(env_module, "set_target_velocity"):
        target_vx = getattr(args, "target_vx", 1.5)
        env_module.set_target_velocity(target_vx)

    logging.info("Beast statics configured: reward_mode=%s keyframe=%s",
                 reward_mode, keyframe_file)
    _BEAST_CONFIGURED = True


def _try_beast_env(env_id: str, *, args=None):
    """Attempt to load a Beast .so environment. Returns None on failure.

    Uses importlib.import_module (cached in sys.modules) to load the .so,
    then wraps with the inlined BeastGymWrapper for the Gymnasium interface.
    """
    try:
        # 1. Load module via importlib (cached — safe for statics + construction)
        env_module = importlib.import_module(env_id)

        # 2. Configure statics BEFORE construction (OnCreate reads them)
        if args is not None:
            _configure_beast_statics(env_module, args)

        # 3. Wrap with BeastGymWrapper (Gymnasium interface)
        EnvClass = getattr(env_module, env_id)
        env = BeastGymWrapper(EnvClass())
        logging.info("Loaded Beast .so env: %s", env_id)
        return env
    except ImportError:
        return None
    except (FileNotFoundError, ModuleNotFoundError):
        return None
