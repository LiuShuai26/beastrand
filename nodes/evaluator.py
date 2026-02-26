"""
Evaluator — MVP stub

- Periodically queries the inference server with a fake observation to simulate eval
- In a real system, would run deterministic rollouts and log metrics
"""

from __future__ import annotations

import logging
import signal
import time

import numpy as np
from nodes.common import child_logging_setup, child_sig_setup
from nodes.logger import child_attach_logger, log_scalar, flush_logger


def main(ctx, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    logging.info("[evaluator] starting (stub)")

    child_attach_logger(logger_queue)
    log_scalar(run=f"evaluator", tag="nodes/evaluator", value=123.4, step=int(ctx.global_step.value))

    req_q = getattr(ctx, "infer_req_q", None)
    resp_q = getattr(ctx, "infer_resp_q", None)
    detached = req_q is None or resp_q is None
    try:
        last = 0.0
        while not ctx.stop_event.is_set():
            time.sleep(0.2)
            if detached:
                continue
            now = time.time()
            if now - last < 2.0:
                continue
            last = now
            rid = int(time.time() * 1e6)
            req = {
                "id": rid,
                "op": "ACT",
                "obs": np.zeros((4,), dtype=np.float32),
                "deterministic": True,
                "actor_id": -1,
                "act_type": "discrete",
                "act_n": 2,
            }
            try:
                req_q.put(req)
                _ = resp_q.get(timeout=0.2)
                logging.info("[evaluator] ran a stub eval step")
            except Exception:
                pass
    finally:
        logging.info("[evaluator] exiting")
