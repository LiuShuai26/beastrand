# projects/pong/make_env.py
"""Pong self-play environment factory using PettingZoo Atari.

Mirrors projects/tennis/make_env.py with pong_v3 instead of tennis_v3.
Preprocessing (via supersuit, applied to all agents uniformly):
  frame_skip(4) → max_observation(2) → grayscale → resize(84×84)
  → frame_stack(4) → reshape(4,84,84)

Final obs shape: (4, 84, 84) uint8 — compatible with AtariPolicy / NatureCNN.
The wrapper (PettingZooParallelWrapper) is shared with the tennis project.

Wrapper is constructed with ``mirror_h_for_opponents=True``: pong_v3 hands the
SAME pixel buffer to both first_0 and second_0, with no agent-id signal. A
single shared self-play policy cannot disambiguate which paddle it controls
from identical input, and collapses to symmetric play (reward stuck near 0).
Horizontally mirroring the opponent's view lets one policy learn a single
"my paddle on the left" perspective and be deployed on either paddle.
"""
from __future__ import annotations

from typing import Optional

import ale_py  # noqa: F401 — registers ALE envs in Gymnasium for the bot path
import gymnasium as gym
import supersuit as ss
from pettingzoo.atari import pong_v3

from projects.pong.dual_env import DualEnvWrapper
from projects.tennis.pettingzoo_wrapper import PettingZooParallelWrapper


def _make_pong_v3(render_mode: Optional[str], frame_stack: int) -> PettingZooParallelWrapper:
    par_env = pong_v3.parallel_env(render_mode=render_mode)
    par_env = ss.frame_skip_v0(par_env, 4)
    par_env = ss.max_observation_v0(par_env, 2)
    par_env = ss.color_reduction_v0(par_env, mode="full")
    par_env = ss.resize_v1(par_env, x_size=84, y_size=84)
    par_env = ss.frame_stack_v2(par_env, stack_size=frame_stack)
    par_env = ss.reshape_v0(par_env, (frame_stack, 84, 84))
    return PettingZooParallelWrapper(par_env, mirror_h_for_opponents=True)


def _make_atari_pong(frame_stack: int) -> gym.Env:
    """Single-agent PongNoFrameskip-v4 with the Atari preprocessing chain used
    by the vs-bot baseline. ``terminal_on_life_loss=False`` so episode semantics
    (one full game) align with pong_v3 — keeps reward / length scales comparable
    when bot episodes are interleaved with self-play episodes in the same run.
    """
    env = gym.make("PongNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    return env


def make_env(
    env_id: str, seed: int = 0, render_mode: Optional[str] = None, *, args=None
):
    """Create a Pong dual-env wrapper: pong_v3 self-play + atari vs-bot.

    The returned wrapper exposes a unified interface; the active underlying env
    is chosen each episode via ``env.set_opponent_mode(mode)`` from the worker.

    Args:
        env_id: Ignored (always creates Pong). Kept for interface compatibility.
        seed: RNG seed.
        render_mode: Optional render mode (e.g. "human", "rgb_array").
        args: Config dataclass (reads ``frame_stack``).
    """
    frame_stack = getattr(args, "frame_stack", 4) if args else 4
    pong = _make_pong_v3(render_mode, frame_stack)
    atari = _make_atari_pong(frame_stack)
    return DualEnvWrapper(pong, atari)
