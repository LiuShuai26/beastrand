# projects/pong/train.py
"""Pong self-play training entrypoint.

Usage:
    python -m projects.pong.train
    python -m projects.pong.train --seed 42
    python -m projects.pong.train --inference-device cuda --learner-device cuda
"""
from __future__ import annotations

import logging
from typing import Optional

import tyro
from projects.pong.config import Args
from beastrand.core.common import setup_logging, set_start_method
from beastrand.nodes.manager import Manager


def main(argv: Optional[list[str]] = None) -> None:
    set_start_method()

    args = tyro.cli(Args, args=argv)
    args.validate()

    run_name = args.make_run_name()
    setup_logging(args.logdir, run_name)

    logging.info("Pong Self-Play Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
