"""MuJoCo PPO training entrypoint.

Usage:
    python -m projects.mujoco.train
    python -m projects.mujoco.train --env-id Humanoid-v4
    python -m projects.mujoco.train --env-id Ant-v4 --seed 42
"""

from __future__ import annotations

import logging
from typing import Optional

import tyro
from beastrand.projects.mujoco.config import Args
from beastrand.core.common import setup_logging, set_start_method
from beastrand.nodes.manager import Manager


def main(argv: Optional[list[str]] = None) -> None:
    set_start_method()

    args = tyro.cli(Args, args=argv)
    args.validate()

    run_name = args.make_run_name()
    setup_logging(args.logdir, run_name)

    logging.info("MuJoCo PPO Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
