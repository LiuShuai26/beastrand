# projects/tennis/pettingzoo_wrapper.py
"""PettingZoo ParallelEnv → Gymnasium Env with multi-agent side-channel.

Exposes standard Gymnasium API for agent 0 (the training agent). Other agents'
I/O goes through ``agent_obs`` / ``agent_actions`` attributes — the same protocol
that RolloutWorker expects.

No auto-reset on termination: the caller (RolloutWorker) handles episode resets.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class PettingZooParallelWrapper(gym.Env):
    """Wraps a PettingZoo ParallelEnv as a Gymnasium env with side-channel protocol.

    Agent 0 gets the standard Gym interface (observation_space, action_space, step,
    reset). Other agents' observations and actions are accessed via:

        env.agent_obs[a]       — observation for agent a (set after reset/step)
        env.agent_actions[a]   — action for agent a (set by worker before step)
    """

    def __init__(self, par_env):
        super().__init__()
        self._env = par_env
        self._agents = list(par_env.possible_agents)
        self.num_agents = len(self._agents)

        # Expose agent 0's spaces as the "official" spaces (for probe_env)
        self.observation_space = par_env.observation_space(self._agents[0])
        self.action_space = par_env.action_space(self._agents[0])

        # Side-channel: per-agent obs and actions
        self.agent_obs: List[Optional[np.ndarray]] = [None] * self.num_agents
        self.agent_actions: List[Optional[np.ndarray]] = [None] * self.num_agents

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        for i, agent in enumerate(self._agents):
            self.agent_obs[i] = obs_dict.get(agent)
        return self.agent_obs[0], info_dict.get(self._agents[0], {})

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step all agents. ``action`` is agent 0's action.

        Other agents' actions must be set in ``self.agent_actions[1:]`` before calling.
        Returns terminal observation on done — caller is responsible for reset.
        """
        self.agent_actions[0] = action

        actions_dict = {}
        for i, agent in enumerate(self._agents):
            act = self.agent_actions[i]
            if act is None:
                act = self._env.action_space(agent).sample()
            actions_dict[agent] = act

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(
            actions_dict
        )

        agent0 = self._agents[0]

        # Update side-channel obs for all agents
        for i, agent in enumerate(self._agents):
            self.agent_obs[i] = obs_dict.get(agent)

        any_term = any(term_dict.values())
        any_trunc = any(trunc_dict.values())

        return (
            self.agent_obs[0],
            float(rew_dict.get(agent0, 0.0)),
            any_term,
            any_trunc,
            info_dict.get(agent0, {}),
        )

    def close(self):
        self._env.close()
