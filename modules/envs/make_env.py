# modules/envs/make_env.py
"""
Minimal gymnasium environment factory.

NOTE: NormalizeObservation and NormalizeReward wrappers maintain per-process
running statistics. In multi-worker training, each worker will have independent
normalization stats, which can cause observation distribution divergence. For
custom environments or situations where this matters, use a custom env factory
(via --make-env-path) without normalization wrappers. The AMP env factory
(make_env_amp.py) does not apply normalization.
"""

from __future__ import annotations
import logging
import gymnasium as gym
import numpy as np

_NORM_WARNED = False


def make_env(env_id: str, seed: int = 0, render_mode: str | None = None,
             *, args=None):
    """
    Create a Gymnasium environment with reproducible seeding.

    WARNING: NormalizeObservation and NormalizeReward use per-process running
    stats. In distributed training with multiple workers, each worker's stats
    will diverge. Consider using make_env_amp or a custom factory for
    environments that require consistent normalization.

    Args:
        env_id: Gym environment ID, e.g. "CartPole-v1"
        seed: RNG seed for env, action space, numpy
        render_mode: Optional render mode (e.g. "human", "rgb_array")

    Returns:
        env: Gymnasium environment instance
    """
    global _NORM_WARNED
    if not _NORM_WARNED:
        logging.warning(
            "make_env: using per-process NormalizeObservation/NormalizeReward. "
            "Stats will diverge across workers. For production training, "
            "consider a custom env factory without normalization wrappers."
        )
        _NORM_WARNED = True

    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
    env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    # Seed env, action space, and numpy RNG
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    return env
