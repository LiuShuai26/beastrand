"""PPO self-play training entrypoint."""
from __future__ import annotations

import logging
from typing import Optional

import tyro
from projects.ppo_selfplay.config import Args
from beastrand.core.common import setup_logging, set_start_method
from beastrand.nodes.manager import Manager


def main(argv: Optional[list[str]] = None) -> None:
    set_start_method()

    args = tyro.cli(Args, args=argv)
    args.validate()

    run_name = args.make_run_name()
    setup_logging(args.logdir, run_name)

    logging.info("PPO Self-Play Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
