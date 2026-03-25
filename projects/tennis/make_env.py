# projects/tennis/make_env.py
"""Tennis self-play environment factory using PettingZoo Atari.

Preprocessing (via supersuit, applied to all agents uniformly):
  frame_skip(4) → max_observation(2) → grayscale → resize(84×84)
  → frame_stack(4) → reshape(4,84,84)

Final obs shape: (4, 84, 84) uint8 — compatible with AtariPolicy / NatureCNN.
"""
from __future__ import annotations

from typing import Optional

import supersuit as ss
from pettingzoo.atari import tennis_v3

from projects.tennis.pettingzoo_wrapper import PettingZooParallelWrapper


def make_env(
    env_id: str, seed: int = 0, render_mode: Optional[str] = None, *, args=None
) -> PettingZooParallelWrapper:
    """Create a Tennis self-play environment with standard Atari preprocessing.

    Args:
        env_id: Ignored (always creates Tennis). Kept for interface compatibility.
        seed: RNG seed.
        render_mode: Optional render mode (e.g. "human", "rgb_array").
        args: Config dataclass (reads ``frame_stack``).
    """
    frame_stack = getattr(args, "frame_stack", 4) if args else 4

    par_env = tennis_v3.parallel_env(render_mode=render_mode)

    # Standard Atari preprocessing via supersuit (applied to all agents)
    par_env = ss.frame_skip_v0(par_env, 4)
    par_env = ss.max_observation_v0(par_env, 2)
    par_env = ss.color_reduction_v0(par_env, mode="full")
    par_env = ss.resize_v1(par_env, x_size=84, y_size=84)
    par_env = ss.frame_stack_v2(par_env, stack_size=frame_stack)
    par_env = ss.reshape_v0(par_env, (frame_stack, 84, 84))

    return PettingZooParallelWrapper(par_env)
