"""
MVP RL — unified entrypoint for PPO training (scaffold)

This script boots the whole framework (actors, learner, evaluator, param server, etc.)
Even if those modules are not yet implemented, it will launch **stub** processes so you
can validate orchestration, CLI, logging, and graceful shutdown.

Once you implement real modules under `nodes/` and `services/`, this entrypoint will
automatically import and launch them instead of stubs.
"""

from __future__ import annotations

import os
import sys
import logging
import multiprocessing as mp
from typing import Optional

import torch
import tyro  # Preferred by the project
from run.run_ppo.ppo_config import Args

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
    # 'spawn' is safest across platforms and for CUDA contexts
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set
    # Use file_system sharing to avoid fd limits with shared tensors
    torch.multiprocessing.set_sharing_strategy("file_system")


def main(argv: Optional[list[str]] = None) -> None:
    _set_start_method()

    args = tyro.cli(Args, args=argv)  # single source of truth
    args.validate()

    run_name = args.make_run_name()

    _setup_logging(args.logdir, run_name)

    logging.info("Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
