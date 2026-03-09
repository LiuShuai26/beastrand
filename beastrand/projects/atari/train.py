"""Atari PPO training entrypoint.

Usage:
    python -m projects.atari.train
    python -m projects.atari.train --env-id PongNoFrameskip-v4
    python -m projects.atari.train --env-id SpaceInvadersNoFrameskip-v4 --seed 42
"""

from __future__ import annotations

import logging
from typing import Optional

import tyro
from beastrand.projects.atari.config import Args
from beastrand.core.common import setup_logging, set_start_method
from beastrand.nodes.manager import Manager


def main(argv: Optional[list[str]] = None) -> None:
    set_start_method()

    args = tyro.cli(Args, args=argv)
    args.validate()

    run_name = args.make_run_name()
    setup_logging(args.logdir, run_name)

    logging.info("Atari PPO Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
