# projects/ppo_selfplay/make_env.py
"""Multi-agent env wrapper and factory for self-play training.

The MultiAgentWrapper presents a standard Gymnasium API for agent 0 (the training
agent), making it compatible with probe_env and existing Gym wrappers. Other agents'
observations and actions are accessed via side-channel attributes:

    env.agent_obs[a]       — observation for agent a (set after reset/step)
    env.agent_actions[a]   — action for agent a (set by worker before step)

The initial implementation wraps N independent copies of the same Gym env. This is
sufficient for testing the framework pipeline. A real self-play env (e.g., two
humanoids fighting in the same Beast scene) replaces the inner env.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class MultiAgentWrapper(gym.Env):
    """Wraps N independent Gym envs as a multi-agent env with side-channel protocol.

    Standard Gymnasium interface is for agent 0 only. Other agents' I/O goes through
    ``agent_obs`` (read) and ``agent_actions`` (write) attributes.

    No auto-reset on termination: the caller (RolloutWorker) handles episode resets.

    This initial implementation runs N independent envs (no physical interaction).
    Replace ``_inner_reset`` / ``_inner_step`` for envs where agents interact.
    """

    def __init__(self, envs: List[gym.Env]):
        assert len(envs) >= 1
        self._envs = envs
        self.num_agents = len(envs)

        # Expose agent 0's spaces as the "official" spaces (for probe_env)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

        # Side-channel: per-agent obs and actions
        self.agent_obs: List[Optional[np.ndarray]] = [None] * self.num_agents
        self.agent_actions: List[Optional[np.ndarray]] = [None] * self.num_agents

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        infos: List[dict] = []
        for i, env in enumerate(self._envs):
            s = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=s, options=options)
            self.agent_obs[i] = obs
            infos.append(info)
        return self.agent_obs[0], infos[0]

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step all agents. ``action`` is agent 0's action.

        Other agents' actions must be set in ``self.agent_actions[1:]`` before calling.
        """
        self.agent_actions[0] = action

        results = []
        for i, env in enumerate(self._envs):
            act = self.agent_actions[i]
            if act is None:
                act = env.action_space.sample()  # fallback for unset actions (e.g., warmup)
            results.append(env.step(act))

        # Unpack agent 0 result as the "official" return
        obs_0, rew_0, term_0, trunc_0, info_0 = results[0]
        self.agent_obs[0] = obs_0

        # Update side-channel obs for other agents
        for i in range(1, self.num_agents):
            self.agent_obs[i] = results[i][0]

        # Any agent terminated → episode is over for all
        any_term = any(r[2] for r in results)
        any_trunc = any(r[3] for r in results)

        return self.agent_obs[0], float(rew_0), any_term, any_trunc, info_0

    def close(self):
        for env in self._envs:
            env.close()


def make_env_selfplay(
    env_id: str, seed: int = 0, render_mode: Optional[str] = None, *, args=None
) -> MultiAgentWrapper:
    """Create a multi-agent self-play environment.

    For now: N independent Gym envs wrapped in MultiAgentWrapper.
    Future: Beast .so multi-agent env.
    """
    num_agents = getattr(args, "num_agents", 2) if args else 2

    envs = []
    for i in range(num_agents):
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        env.reset(seed=seed + i)
        envs.append(env)

    return MultiAgentWrapper(envs)
