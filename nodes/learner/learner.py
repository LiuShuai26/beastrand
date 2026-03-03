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
    learning_starts = getattr(args, "learning_starts", batch_size)

    batch_buf = BatchBuffer(meta=meta, batch_size=batch_size, buffer_size=buffer_size)
    buf_lock = threading.Lock()
    have_batch = threading.Condition(buf_lock)

    # ------------------------------------------------------------------
    # Ingest thread: recv filled traj → prepare → append → recycle
    # ------------------------------------------------------------------
    # If the algorithm supports batched finalization (prepare_batch_finalize),
    # accumulate trajectories and flush them together so the discriminator
    # forward pass runs on one large batch instead of N small ones.
    _use_batched_finalize = hasattr(algorithm, "prepare_batch_finalize")
    _batch_trigger = max(1, batch_size // T) if _use_batched_finalize else 1
    _max_policy_lag = getattr(args, "max_policy_lag", 0)

    def _flush_pending(pending):
        """Run batched finalize, append all views, recycle traj buffers."""
        if not pending:
            return

        views = [v for _, v in pending]
        algorithm.prepare_batch_finalize(views)

        with buf_lock:
            for _, v in pending:
                batch_buf.append_slot(v)
            rdy = batch_buf.valid_steps >= learning_starts

        for ti, _ in pending:
            buffer_mgr.traj_buffer_queue.put(ti)

        pending.clear()

        if rdy:
            with have_batch:
                have_batch.notify()

    def _is_stale(view) -> bool:
        """Check if trajectory data is too old to train on."""
        if _max_policy_lag <= 0:
            return False
        if "model_version" not in view:
            return False
        avg_ver = int(np.mean(view["model_version"]))
        lag = int(buffer_mgr.policy_version.item()) - avg_ver
        return lag > _max_policy_lag

    def ingest_worker():
        pending = []  # list of (traj_idx, view_dict)
        _discarded = 0
        _last_discard_log = time.monotonic()

        while not ctx.stop_event.is_set():
            # Poll with timeout so we can check stop_event periodically
            ready_socks = bus.poll(timeout_ms=100)
            if "filled_in" not in ready_socks:
                continue

            # Drain all available filled traj IDs at once
            try:
                raw_msgs = bus.recv_many("filled_in")
            except Exception:
                continue

            notify_needed = False
            for raw in raw_msgs:
                traj_idx = struct.unpack("<i", raw)[0]

                # Get numpy views into this trajectory (zero-copy)
                view = buffer_mgr.slot_as_numpy(traj_idx)

                # Discard stale trajectories (backpressure)
                if _is_stale(view):
                    buffer_mgr.traj_buffer_queue.put(traj_idx)
                    _discarded += 1
                    continue

                # Phase 1: lightweight per-traj prep (numpy only for batched path,
                # or full GAE computation for the legacy path)
                algorithm.prepare_batch(view)

                if _use_batched_finalize:
                    # Accumulate until batch_trigger, then flush all at once
                    pending.append((traj_idx, view))
                    if len(pending) >= _batch_trigger:
                        _flush_pending(pending)
                else:
                    # Plain PPO: append + recycle immediately
                    with buf_lock:
                        batch_buf.append_slot(view)
                        if batch_buf.valid_steps >= learning_starts:
                            notify_needed = True

                    buffer_mgr.traj_buffer_queue.put(traj_idx)

            if notify_needed:
                with have_batch:
                    have_batch.notify()

            # Log discard stats periodically
            now = time.monotonic()
            if _discarded > 0 and now - _last_discard_log > 10.0:
                logging.info("[learner] discarded %d stale trajectories (max_policy_lag=%d)",
                             _discarded, _max_policy_lag)
                _discarded = 0
                _last_discard_log = now

        # Flush any remaining trajectories on shutdown so traj buffers
        # are recycled and workers blocked on traj_queue.get() unblock.
        if pending:
            _flush_pending(pending)

    ing_thread = threading.Thread(target=ingest_worker, name="ingest", daemon=True)
    ing_thread.start()

    record_cls = get_object_from_path(args.data_record_path)

    logging.info("[learner] ready (device=%s, batch_size=%d, T=%d)", device, batch_size, T)

    last_training_time = time.perf_counter()

    # --- Periodic checkpoint ---
    import os
    _ckpt_interval = getattr(args, "checkpoint_interval", 0)
    _ckpt_version_interval = max(1, _ckpt_interval // batch_size) if _ckpt_interval > 0 else 0
    _last_ckpt_version = 0
    _save_dir = os.path.join(getattr(args, "logdir", "train_logs"), ctx.run_name)

    def _save_checkpoint(version: int) -> None:
        if not hasattr(algorithm, "save_checkpoint"):
            return
        try:
            algorithm.save_checkpoint(_save_dir, policy)
            logging.info("[learner] periodic checkpoint saved (version=%d)", version)
        except Exception:
            logging.exception("[learner] periodic checkpoint failed")

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

            batch = record_cls.build_batch(ctx, view)

            # Policy lag (for logging only — stale data is discarded in ingest)
            if "model_version" in view:
                avg_version = int(np.mean(view["model_version"]))
                policy_lag = int(buffer_mgr.policy_version.item()) - avg_version
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

            # --- Periodic checkpoint ---
            if _ckpt_version_interval > 0 and version - _last_ckpt_version >= _ckpt_version_interval:
                _save_checkpoint(version)
                _last_ckpt_version = version

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
        # Wait for ingest thread to flush any pending traj buffers so that
        # workers blocked on traj_queue.get() are unblocked before we exit.
        ing_thread.join(timeout=5.0)

        # Save final checkpoint on exit (if algorithm supports it)
        _save_checkpoint(int(buffer_mgr.policy_version.item()))

        bus.close_all()
        logging.info("[learner] exiting")
