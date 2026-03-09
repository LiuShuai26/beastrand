"""
Learner (v2): ingest thread + training loop.

Changes from v1:
  - Reads trajectory data from BufferMgr shared tensors (zero-copy numpy views)
  - Uses ParameterServer for weight sync (no PUB/SUB, no ParamServer process)
  - Releases traj slots via split_flags[flat_idx, split_idx] = 0 (replaces mp.Queue)
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

from beastrand.utils.import_utils import get_object_from_path

from beastrand.nodes.common import child_logging_setup, child_sig_setup
from beastrand.nodes.logger import child_attach_logger, log_scalar
from beastrand.nodes.learner.batch_buffer import BatchBuffer
from beastrand.strandbus.strandbus import StrandBus


def main(ctx, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[learner] starting")

    args = ctx.args
    buffer_mgr = ctx.buffer_mgr

    # --- Device ---
    device = torch.device(args.learner_device)

    # --- Policy + Optimizer + Algorithm ---
    policy_cls = get_object_from_path(args.policy_path)
    policy = policy_cls(ctx).to(device)
    opt = policy.build_optimizers(ctx)

    AlgorithmClass = get_object_from_path(args.algorithm_path)
    algorithm = AlgorithmClass(ctx, policy, opt, device)

    # --- Weight sharing: use ParameterServer created by Manager ---
    param_server = ctx.param_server
    # Load initial shared weights into our policy so we start consistent
    from beastrand.utils.model_sharing import _load_state_into_model
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

    _split_depth = buffer_mgr.split_depth
    _num_envs = buffer_mgr.num_envs_per_worker

    def _release_slot(traj_idx: int) -> None:
        """Release a trajectory slot back to its worker via split_flags."""
        flat_idx = traj_idx // _split_depth
        split_idx = traj_idx % _split_depth
        buffer_mgr.split_flags[flat_idx, split_idx] = 0

    def _flush_pending(pending):
        """Run batched finalize, append all views, release traj slots."""
        if not pending:
            return

        views = [v for _, v in pending]
        algorithm.prepare_batch_finalize(views)

        with buf_lock:
            for _, v in pending:
                batch_buf.append_slot(v)
            rdy = batch_buf.valid_steps >= learning_starts

        for ti, _ in pending:
            _release_slot(ti)

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

    # Ingest-thread profiling state (shared via closure).
    _ingest_gae_total = 0.0    # cumulative prepare_batch time (seconds)
    _ingest_traj_count = 0     # trajectories processed
    _ingest_last_log = time.monotonic()
    _ingest_log_interval = 10.0  # seconds between ingest profile logs

    def ingest_worker():
        nonlocal _ingest_gae_total, _ingest_traj_count, _ingest_last_log
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
                    _release_slot(traj_idx)
                    _discarded += 1
                    continue

                # Phase 1: lightweight per-traj prep (numpy only for batched path,
                # or full GAE computation for the legacy path)
                _t_gae = time.monotonic()
                algorithm.prepare_batch(view)
                _ingest_gae_total += time.monotonic() - _t_gae
                _ingest_traj_count += 1

                if _use_batched_finalize:
                    # Accumulate until batch_trigger, then flush all at once
                    pending.append((traj_idx, view))
                    if len(pending) >= _batch_trigger:
                        _flush_pending(pending)
                else:
                    # Plain PPO: append + release immediately
                    with buf_lock:
                        batch_buf.append_slot(view)
                        if batch_buf.valid_steps >= learning_starts:
                            notify_needed = True

                    _release_slot(traj_idx)

            if notify_needed:
                with have_batch:
                    have_batch.notify()

            # Log discard + ingest profile stats periodically
            now = time.monotonic()
            if _discarded > 0 and now - _last_discard_log > 10.0:
                logging.info("[learner] discarded %d stale trajectories (max_policy_lag=%d)",
                             _discarded, _max_policy_lag)
                _discarded = 0
                _last_discard_log = now
            if _ingest_traj_count > 0 and now - _ingest_last_log >= _ingest_log_interval:
                avg_gae_ms = _ingest_gae_total / _ingest_traj_count * 1000.0
                logging.info(
                    "[learner/ingest] gae=%.2fms/traj  trajs=%d  (last %.0fs)",
                    avg_gae_ms, _ingest_traj_count, _ingest_log_interval,
                )
                _ingest_gae_total = 0.0
                _ingest_traj_count = 0
                _ingest_last_log = now

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
            # ---- 1. Wait for data ----------------------------------------
            t0 = time.perf_counter()
            with have_batch:
                ready = have_batch.wait_for(
                    lambda: batch_buf.valid_steps >= learning_starts or ctx.stop_event.is_set(),
                    timeout=0.5,
                )

            if ctx.stop_event.is_set():
                break
            if not ready:
                continue

            wait_ms = (time.perf_counter() - t0) * 1000.0

            # ---- 2. Build batch (numpy → tensor) -------------------------
            t1 = time.perf_counter()
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

            build_batch_ms = (time.perf_counter() - t1) * 1000.0

            # ---- 3. Train ------------------------------------------------
            t2 = time.perf_counter()
            stats = algorithm.update(batch)
            update_ms = (time.perf_counter() - t2) * 1000.0

            # ---- 4. Sync weights -----------------------------------------
            t3 = time.perf_counter()
            version = param_server.update(policy)
            weight_sync_ms = (time.perf_counter() - t3) * 1000.0

            total_ms = (time.perf_counter() - last_training_time) * 1000.0
            last_training_time = time.perf_counter()

            # --- Periodic checkpoint ---
            if _ckpt_version_interval > 0 and version - _last_ckpt_version >= _ckpt_version_interval:
                _save_checkpoint(version)
                _last_ckpt_version = version

            # --- Logging ---
            log_scalar(run="learner", tag="policy_lag", value=policy_lag, step=version)
            log_scalar(run="learner", tag="wait_for_data_ms", value=wait_ms, step=version)
            log_scalar(run="learner", tag="build_batch_ms", value=build_batch_ms, step=version)
            log_scalar(run="learner", tag="update_ms", value=update_ms, step=version)
            log_scalar(run="learner", tag="weight_sync_ms", value=weight_sync_ms, step=version)
            log_scalar(run="learner", tag="total_iter_ms", value=total_ms, step=version)
            for k, v in stats.items():
                try:
                    log_scalar(run="learner", tag=k, value=float(v), step=version)
                except Exception:
                    pass

    finally:
        # Wait for ingest thread to flush pending traj buffers and release
        # split_flags so that workers spin-waiting on them can exit cleanly.
        ing_thread.join(timeout=5.0)

        # Save final checkpoint on exit (if algorithm supports it)
        _save_checkpoint(int(buffer_mgr.policy_version.item()))

        bus.close_all()
        logging.info("[learner] exiting")
