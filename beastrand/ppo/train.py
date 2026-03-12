"""PPO training entrypoint."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Optional

import tyro
from beastrand.ppo.config import Args
from beastrand.core.common import setup_logging, set_start_method
from beastrand.nodes.manager import Manager


def _run_name_from_resume(resume: str, logdir: str) -> Optional[str]:
    """Extract run name from a checkpoint path like <logdir>/<run_name>/checkpoints/ckpt_step_N.pt."""
    resume = os.path.normpath(resume)
    logdir = os.path.normpath(logdir)
    if resume.startswith(logdir + os.sep):
        rel = resume[len(logdir) + 1:]
        parts = rel.split(os.sep)
        if parts:
            return parts[0]
    # Fallback: grandparent directory of the checkpoint file
    return os.path.basename(os.path.dirname(os.path.dirname(resume))) or None


def main(argv: Optional[list[str]] = None) -> None:
    set_start_method()

    args = tyro.cli(Args, args=argv)
    args.validate()

    run_name = args.make_run_name()
    # On resume without an explicit --run_name, reuse the original run directory
    if args.resume and not args.run_name:
        extracted = _run_name_from_resume(args.resume, args.logdir)
        if extracted:
            run_name = extracted
            args.run_name = run_name  # propagate to Manager

    setup_logging(args.logdir, run_name)

    # Save config to disk for reproducibility / resume inspection
    _cfg_dir = os.path.join(args.logdir, run_name)
    os.makedirs(_cfg_dir, exist_ok=True)
    _cfg_path = os.path.join(_cfg_dir, "config.json")
    with open(_cfg_path, "w") as _f:
        json.dump(dataclasses.asdict(args), _f, indent=2, default=str)

    logging.info("Args: %s", args)

    mgr = Manager(args)
    mgr.launch(wandb_config=dataclasses.asdict(args))
    mgr.run_until_complete()


if __name__ == "__main__":
    main()
