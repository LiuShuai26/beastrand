"""
DataServer (v2): pure filled-trajectory-ID forwarder.

Workers PUSH filled traj_idx → DataServer → PUSH to Learner.
Free slot recycling goes through BufferMgr.traj_buffer_queue (mp.Queue),
not through DataServer.

Kept as a separate node for future extensibility (e.g., multi-policy
routing for self-play).
"""
from __future__ import annotations

import logging
import zmq

from nodes.common import child_logging_setup, child_sig_setup
from nodes.logger import child_attach_logger
from strandbus.strandbus import StrandBus


class DataServer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.bus = StrandBus()

        base = ctx.ipc_dir
        # filled traj IDs: workers → this → learner
        self.bus.open("filled_in", mode="pull", endpoint=f"{base}/data.filled.in", bind=True)
        self.bus.open("filled_out", mode="push", endpoint=f"{base}/data.filled.out", bind=True)

    def serve(self) -> None:
        logging.info("[data_server] ready (pure forwarder)")
        try:
            while not self.ctx.stop_event.is_set():
                # Block for first message, then drain
                try:
                    msgs = self.bus.recv_many("filled_in")
                except Exception:
                    continue

                for msg in msgs:
                    try:
                        self.bus.send("filled_out", msg)
                    except Exception as e:
                        logging.error("[data_server] forward failed: %s", e)
        finally:
            self.bus.close_all()
            logging.info("[data_server] exiting")


def main(ctx, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[data_server] starting")
    DataServer(ctx).serve()
    logging.info("[data_server] stopped")
