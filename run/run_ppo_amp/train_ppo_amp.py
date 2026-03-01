"""
PPO-AMP training entrypoint.

Boots the distributed beatstrand framework with PPO-AMP modules:
  - PPOAMPPolicy (dual value heads)
  - PPOAMPAlgorithm (discriminator + multi-group GAE)
  - PPOAMPDataRecord (transition pairs + per-group returns)

Usage:
    python -m run.run_ppo_amp.train_ppo_amp --keyframe-file path/to/keyframes.json
    python -m run.run_ppo_amp.train_ppo_amp --env-id Humanoid-v5 --keyframe-file kf.json --num-workers 16
"""

from __future__ import annotations

import os
import sys
import logging
import multiprocessing as mp
from typing import Optional

import torch
import tyro
from run.run_ppo_amp.ppo_amp_config import Args

from nodes.manager import Manager


def _setup_logging(logdir: str, run_name: str) -> None:
    os.makedirs(f"{logdir}/{run_name}", exist_ok=True)
    logfile = os.path.join(f"{logdir}/{run_name}", f"{run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    logging.info("Logging to %s", logfile)


def _set_start_method() -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")


def _configure_from_so(args: Args) -> None:
    """Read amp_obs_slices from the Beast .so module (single source of truth)."""
    try:
        import importlib
        env_module = importlib.import_module(args.env_id)
        if hasattr(env_module, "set_brain_class"):
            env_module.set_brain_class(args.brain_class)
        if hasattr(env_module, "amp_obs_slices"):
            args.amp_obs_slices = [tuple(s) for s in env_module.amp_obs_slices()]
            args.amp_obs_dim = sum(e - s for s, e in args.amp_obs_slices)
    except (ImportError, ModuleNotFoundError):
        pass  # Not a Beast env, use config defaults


def main(argv: Optional[list[str]] = None) -> None:
    _set_start_method()

    args = tyro.cli(Args, args=argv)
    _configure_from_so(args)
    args.validate()

    run_name = args.make_run_name()

    _setup_logging(args.logdir, run_name)

    logging.info("PPO-AMP Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
