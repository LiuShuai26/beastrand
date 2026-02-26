"""
Learner (v2): ingest thread + training loop.

Changes from v1:
  - Reads trajectory data from BufferMgr shared tensors (zero-copy numpy views)
  - Uses ParameterServer for weight sync (no PUB/SUB, no ParamServer process)
  - Recycles traj indices via BufferMgr.traj_buffer_queue (mp.Queue)
  - Receives filled traj_idx as struct.pack("<i") from DataServer
"""
from __future__ import annotations

import logging
import struct
import time
import threading
from typing import Any, Dict

import numpy as np
import torch

from utils.import_utils import get_object_from_path

from nodes.common import child_logging_setup, child_sig_setup
from nodes.logger import child_attach_logger, log_scalar
from nodes.learner.batch_buffer import BatchBuffer
from strandbus.strandbus import StrandBus


def main(ctx, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[learner] starting")

    args = ctx.args
    buffer_mgr = ctx.buffer_mgr

    # --- Device ---
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        try:
            device = torch.device("cuda", getattr(args, "gpu_id", 0))
        except Exception:
            device = torch.device("cpu")

    # --- Policy + Optimizer + Algorithm ---
    policy_cls = get_object_from_path(args.policy_path)
    policy = policy_cls(ctx).to(device)
    opt = policy.build_optimizers(ctx)

    AlgorithmClass = get_object_from_path(args.algorithm_path)
    algorithm = AlgorithmClass(ctx, policy, opt, device)

    # --- Weight sharing: use ParameterServer created by Manager ---
    param_server = ctx.param_server
    # Load initial shared weights into our policy so we start consistent
    from utils.model_sharing import _load_state_into_model
    _load_state_into_model(param_server.shared_state, policy)
    # Immediately push our (possibly re-initialized) weights back
    param_server.update(policy)

    # --- ZMQ (only filled_in for receiving filled traj IDs) ---
    bus = StrandBus()
    base = ctx.ipc_dir
    bus.open("filled_in", mode="pull", endpoint=f"{base}/data.filled.out", bind=False)

    # --- Batch buffer ---
    T = args.rollout
    schema = buffer_mgr.schema()
    meta = {"schema": schema, "T": T}
    batch_size = args.batch_size
    buffer_size = getattr(args, "replay_capacity", batch_size)
    buffer_mode = getattr(args, "buffer_mode", "fullpass")
    learning_starts = getattr(args, "learning_starts", batch_size)

    batch_buf = BatchBuffer(meta=meta, buffer_mode=buffer_mode,
                            batch_size=batch_size, buffer_size=buffer_size)
    buf_lock = threading.Lock()
    have_batch = threading.Condition(buf_lock)

    # ------------------------------------------------------------------
    # Ingest thread: recv filled traj → GAE → append → recycle
    # ------------------------------------------------------------------
    def ingest_worker():
        while not ctx.stop_event.is_set():
            try:
                raw = bus.recv("filled_in", noblock=False)
            except Exception:
                continue

            traj_idx = struct.unpack("<i", raw)[0]

            # Get numpy views into this trajectory (zero-copy)
            view = buffer_mgr.slot_as_numpy(traj_idx)

            # Compute GAE / prepare batch fields
            algorithm.prepare_batch(view)

            with buf_lock:
                batch_buf.append_slot(view)
                ready = batch_buf.valid_steps >= learning_starts

            # Recycle trajectory buffer back to queue
            buffer_mgr.traj_buffer_queue.put(traj_idx)

            if ready:
                with have_batch:
                    have_batch.notify()

    ing_thread = threading.Thread(target=ingest_worker, name="ingest", daemon=True)
    ing_thread.start()

    logging.info("[learner] ready (device=%s, batch_size=%d, T=%d)", device, batch_size, T)

    last_training_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    try:
        while not ctx.stop_event.is_set():
            start_get_data = time.perf_counter()

            with have_batch:
                ready = have_batch.wait_for(
                    lambda: batch_buf.valid_steps >= learning_starts or ctx.stop_event.is_set(),
                    timeout=0.5,
                )

            if ctx.stop_event.is_set():
                break
            if not ready:
                continue

            with buf_lock:
                view = batch_buf.get_batch()

            N = view.get("reward", np.empty(0)).shape[0]
            if N == 0:
                continue

            record_cls = get_object_from_path(args.data_record_path)
            batch = record_cls.build_batch(ctx, view)

            # Policy lag check
            if "model_version" in view:
                avg_version = int(np.mean(view["model_version"]))
                policy_lag = int(buffer_mgr.policy_version.item()) - avg_version
                max_lag = getattr(args, "max_policy_lag", getattr(args, "policy_lag", 100))
                if getattr(args, "lag_controll", False) and policy_lag > max_lag:
                    time.sleep(0.1)
                    continue
            else:
                policy_lag = 0

            end_get_data = time.perf_counter()
            get_data_time = (end_get_data - start_get_data) * 1000.0

            # --- Train ---
            stats = algorithm.update(batch)
            end_update = time.perf_counter()
            train_batch_time = (end_update - end_get_data) * 1000.0
            training_time = (end_update - last_training_time) * 1000.0
            last_training_time = end_update

            # --- Update shared weights ---
            version = param_server.update(policy)

            # --- Logging ---
            log_scalar(run="learner", tag="policy_lag", value=policy_lag, step=version)
            log_scalar(run="learner", tag="get_data_time_ms", value=get_data_time, step=version)
            log_scalar(run="learner", tag="train_batch_time_ms", value=train_batch_time, step=version)
            log_scalar(run="learner", tag="total_train_time_ms", value=training_time, step=version)
            for k, v in stats.items():
                try:
                    log_scalar(run="learner", tag=k, value=float(v), step=version)
                except Exception:
                    pass

    finally:
        # Save checkpoint on exit (if algorithm supports it)
        if hasattr(algorithm, "save_checkpoint"):
            import os
            save_dir = os.path.join(
                getattr(args, "logdir", "train_logs"),
                ctx.run_name,
            )
            try:
                algorithm.save_checkpoint(save_dir, policy)
            except Exception:
                logging.exception("[learner] save_checkpoint failed")

        bus.close_all()
        logging.info("[learner] exiting")
