"""Dual-env wrapper for Pong: routes between pong_v3 self-play and PongNoFrameskip-v4 vs ALE scripted bot.

Three self-play experiments (snapshot 20M, mirror, max=200 bigpool) all stalled
at reward 0-4. The dominant remaining hypothesis is self-play cold-start: a
randomly-initialized learner playing snapshots of itself never gets a clear
"how to win" gradient because the opponent is always near its own skill level.
vs-bot Pong learns to +20.5 because the ALE scripted left paddle is a fixed
weak opponent that provides a stable signal.

This wrapper lets a single training run mix BOTH signals: for each episode,
the worker samples an opponent mode and the wrapper routes to the matching
underlying env:

- ``mode in {"snapshot", "latest"}``: active env = pong_v3 (two trainable
  paddles, the existing PettingZooParallelWrapper handles agent_obs / actions
  including the mirror_h_for_opponents=True flip on agent 1).
- ``mode == "bot"``: active env = single-agent PongNoFrameskip-v4 with
  Atari preprocessing. agent_obs[0] is exposed as the H-flipped Atari frame
  (so the policy keeps the "my paddle on the left" perspective it learned in
  mirrored pong_v3 — single-agent player controls the RIGHT paddle, so the
  flip aligns it with pong_v3 first_0's view). agent_obs[1] is None and the
  worker must skip inference / action writes for agent 1.

Action space is Discrete(6) in both envs. Reward is ±1 per scored point in
both. Only UP/DOWN actions matter for paddle control, and these are vertical
— invariant under H-flip — so no action remapping is needed.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np


_VALID_MODES = ("snapshot", "latest", "bot")


class DualEnvWrapper(gym.Env):
    """Gymnasium env wrapping two underlying envs, switchable per-episode.

    The active env is selected by the most recent ``set_opponent_mode`` call;
    the switch takes effect on the next ``reset()``. Until ``set_opponent_mode``
    is called, mode defaults to ``"snapshot"`` so the wrapper behaves identically
    to plain pong_v3 self-play.
    """

    def __init__(self, pong_v3_env: gym.Env, atari_env: gym.Env):
        """
        Args:
            pong_v3_env: A ``PettingZooParallelWrapper`` around a preprocessed
                ``pong_v3`` parallel env (already configured with
                ``mirror_h_for_opponents=True``).
            atari_env: A single-agent Gymnasium env (PongNoFrameskip-v4 with the
                standard Atari preprocessing chain — same final (4,84,84) uint8
                obs shape as the pong_v3 env).
        """
        super().__init__()
        self._pong = pong_v3_env
        self._atari = atari_env
        self._mode: str = "snapshot"
        self._active: gym.Env = self._pong

        # Both envs MUST agree on obs/action space after preprocessing.
        self.observation_space = self._pong.observation_space
        self.action_space = self._pong.action_space
        if self._atari.observation_space.shape != self.observation_space.shape:
            raise ValueError(
                f"atari obs shape {self._atari.observation_space.shape} != "
                f"pong_v3 obs shape {self.observation_space.shape}"
            )
        if self._atari.action_space.n != self.action_space.n:
            raise ValueError(
                f"atari action n {self._atari.action_space.n} != "
                f"pong_v3 action n {self.action_space.n}"
            )

        # Multi-agent contract: num_agents is fixed at 2 for buffer compatibility.
        # In bot mode, agent_obs[1] is None and the worker must short-circuit.
        self.num_agents = 2
        self.agent_obs: List[Optional[np.ndarray]] = [None, None]
        self.agent_actions: List[Optional[np.ndarray]] = [None, None]

    def set_opponent_mode(self, mode: str) -> None:
        """Stage the opponent mode for the NEXT reset(). Takes effect on reset."""
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self._mode = mode

    @property
    def opponent_mode(self) -> str:
        return self._mode

    def _hflip(self, obs):
        return np.ascontiguousarray(obs[..., ::-1])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self._mode == "bot":
            self._active = self._atari
            obs, info = self._atari.reset(seed=seed, options=options)
            self.agent_obs[0] = self._hflip(np.asarray(obs))
            self.agent_obs[1] = None
            return self.agent_obs[0], info
        else:
            self._active = self._pong
            obs, info = self._pong.reset(seed=seed, options=options)
            # PettingZooParallelWrapper writes into agent_obs internally; mirror to ours.
            self.agent_obs[0] = self._pong.agent_obs[0]
            self.agent_obs[1] = self._pong.agent_obs[1]
            return self.agent_obs[0], info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._mode == "bot":
            # Single-agent atari handles the ALE scripted opponent internally.
            obs, reward, term, trunc, info = self._atari.step(action)
            self.agent_obs[0] = self._hflip(np.asarray(obs))
            self.agent_obs[1] = None
            return self.agent_obs[0], float(reward), bool(term), bool(trunc), info
        else:
            # pong_v3 path: PettingZooParallelWrapper reads agent_actions[1] internally.
            self._pong.agent_actions[1] = self.agent_actions[1]
            obs, reward, term, trunc, info = self._pong.step(action)
            self.agent_obs[0] = self._pong.agent_obs[0]
            self.agent_obs[1] = self._pong.agent_obs[1]
            return obs, reward, term, trunc, info

    def close(self):
        self._pong.close()
        self._atari.close()
