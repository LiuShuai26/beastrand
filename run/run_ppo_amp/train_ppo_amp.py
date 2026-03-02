"""PPO-AMP training entrypoint."""

from __future__ import annotations

import logging
from typing import Optional

import tyro
from run.run_ppo_amp.ppo_amp_config import Args
from run.common import setup_logging, set_start_method
from nodes.manager import Manager


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
    set_start_method()

    args = tyro.cli(Args, args=argv)
    _configure_from_so(args)
    args.validate()

    run_name = args.make_run_name()
    setup_logging(args.logdir, run_name)

    logging.info("PPO-AMP Args: %s", args)

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
