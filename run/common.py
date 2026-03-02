import logging
import multiprocessing as mp
import os
import sys

import torch


def setup_logging(logdir: str, run_name: str) -> None:
    os.makedirs(f"{logdir}/{run_name}", exist_ok=True)
    logfile = os.path.join(f"{logdir}/{run_name}", f"{run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    logging.info("Logging to %s", logfile)


def set_start_method() -> None:
    """'spawn' is safest across platforms and for CUDA contexts."""
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set
    # Use file_system sharing to avoid fd limits with shared tensors
    torch.multiprocessing.set_sharing_strategy("file_system")
