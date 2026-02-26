# nodes/make_env.py
"""
Minimal gymnasium environment factory.
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int = 0, render_mode: str | None = None):
    """
    Create a Gymnasium environment with reproducible seeding.

    Args:
        env_id: Gym environment ID, e.g. "CartPole-v1"
        seed: RNG seed for env, action space, numpy
        render_mode: Optional render mode (e.g. "human", "rgb_array")

    Returns:
        env: Gymnasium environment instance
    """
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
