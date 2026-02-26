"""
RolloutWorker (v3): async per-env rollout via shared memory flags.

Each worker manages ``num_envs_per_worker`` environments. All envs operate
independently — no splits, no synchronous waiting. The InferenceServer sets
``ready_flags[worker_idx, env_idx] = 1`` after writing action/logp/value into
shared tensors, and the worker polls these flags to advance ready envs.

Worker never blocks: it always has some env to step.
"""
from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List, Optional

import numpy as np
import torch

from modules.envs.make_env import make_env
from nodes.common import child_logging_setup, child_sig_setup
from nodes.logger import child_attach_logger, log_scalar
from strandbus.strandbus import StrandBus

# Must match inference_server.py
REQ_FMT = "<iiiii"
REQ_SIZE = struct.calcsize(REQ_FMT)
OP_ACT = 0
OP_VALUE = 1


@dataclass
class EnvState:
    """Mutable state for a single environment."""
    env: object
    obs: np.ndarray
    traj_idx: int
    env_idx: int
    step: int = 0
    pending: bool = False
    episode_reward: float = 0.0
    episode_length: int = 0
    done: bool = False


class _ProfileAccum:
    """Lightweight cumulative timer for profiling hot loops."""
    __slots__ = ("_timers", "_counts", "_last_report", "_interval")

    def __init__(self, interval: float = 5.0):
        self._timers: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._last_report = time.monotonic()
        self._interval = interval

    def add(self, name: str, dt: float) -> None:
        self._timers[name] = self._timers.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def maybe_report(self, tag: str) -> Optional[str]:
        now = time.monotonic()
        if now - self._last_report < self._interval:
            return None
        elapsed = now - self._last_report
        parts = []
        for k in sorted(self._timers):
            total_ms = self._timers[k] * 1000
            cnt = self._counts[k]
            avg_ms = total_ms / cnt if cnt else 0
            parts.append(f"{k}: {total_ms:.0f}ms/{cnt}calls ({avg_ms:.2f}ms avg)")
        self._timers.clear()
        self._counts.clear()
        self._last_report = now
        if parts:
            return f"[{tag}] {elapsed:.1f}s | " + ", ".join(parts)
        return None


class RolloutWorker:
    def __init__(self, ctx, worker_idx: int):
        self.ctx = ctx
        self.worker_idx = worker_idx
        args = ctx.args

        self.T = args.rollout
        self.num_envs = getattr(args, "num_envs_per_worker", 2)

        self.use_lstm = bool(getattr(args, "use_rnn", False) or getattr(args, "use_lstm", False))
        self.bootstrap_value = bool(getattr(args, "value_bootstrap", False) or getattr(args, "bootstrap_value", False))

        # Shared tensors from BufferMgr
        self.traj_tensors = ctx.buffer_mgr.traj_tensors
        self.traj_queue = ctx.buffer_mgr.traj_buffer_queue
        self.ready_flags = ctx.buffer_mgr.ready_flags

        # Action space info (set by Manager on ctx)
        self.act_discrete = ctx.act_kind == "discrete"

        # Pre-cache frequently used tensors
        self._obs_tensor = self.traj_tensors["obs"]
        self._act_tensor = self.traj_tensors["action"]
        self._rew_tensor = self.traj_tensors["reward"]
        self._done_tensor = self.traj_tensors["done"]

        # --- ZMQ (only for sending requests + filled trajectories) ---
        self.bus = StrandBus()
        base = ctx.ipc_dir
        self.bus.open("infer_req", mode="push", endpoint=f"{base}/infer.req", bind=False)
        self.bus.open("filled_out", mode="push", endpoint=f"{base}/data.filled.in", bind=False)

        # --- Profiling (only worker 0) ---
        self.prof = _ProfileAccum(interval=5.0) if worker_idx == 0 else None

        # --- Resolve env factory (configurable via args.make_env_path) ---
        _make_env_path = getattr(args, "make_env_path", None)
        if _make_env_path:
            from utils.import_utils import get_object_from_path
            _make_env = get_object_from_path(_make_env_path)
        else:
            _make_env = make_env

        # --- Create all environments (flat, no splits) ---
        self.envs: List[EnvState] = []
        for ei in range(self.num_envs):
            seed = args.seed + worker_idx * self.num_envs + ei
            env = _make_env(args.env_id, seed=seed)
            obs, _ = env.reset(seed=seed)
            traj_idx = self.traj_queue.get()
            self.envs.append(EnvState(
                env=env, obs=obs, traj_idx=traj_idx, env_idx=ei,
            ))

    # ------------------------------------------------------------------
    # Core loop — async per-env polling
    # ------------------------------------------------------------------

    def run(self):
        logging.info("[worker:%d] ready (%d envs, async, T=%d)",
                     self.worker_idx, self.num_envs, self.T)

        # Initial: send inference requests for all envs
        for es in self.envs:
            self._send_request(es, OP_ACT)

        try:
            while not self.ctx.stop_event.is_set():
                t0 = time.monotonic()
                stepped_any = False

                for es in self.envs:
                    if not es.pending:
                        continue
                    if not self.ready_flags[self.worker_idx, es.env_idx]:
                        continue

                    # Clear flag and advance
                    self.ready_flags[self.worker_idx, es.env_idx] = 0
                    t1 = time.monotonic()

                    self._advance_single(es)
                    t2 = time.monotonic()

                    self._send_request(es, OP_ACT)
                    t3 = time.monotonic()

                    stepped_any = True

                    if self.prof:
                        self.prof.add("advance", t2 - t1)
                        self.prof.add("send_req", t3 - t2)

                if self.prof:
                    t_end = time.monotonic()
                    if not stepped_any:
                        self.prof.add("idle_spin", t_end - t0)
                    report = self.prof.maybe_report(f"worker:{self.worker_idx}")
                    if report:
                        logging.info(report)

        finally:
            self.bus.close_all()
            logging.info("[worker:%d] exiting", self.worker_idx)

    # ------------------------------------------------------------------
    # Send inference request (single env)
    # ------------------------------------------------------------------

    def _send_request(self, es: EnvState, op: int) -> None:
        # Write obs into shared tensor
        self._obs_tensor[es.traj_idx, es.step] = torch.from_numpy(
            np.asarray(es.obs, dtype=np.float32))

        # Write mask for LSTM
        if self.use_lstm and "mask" in self.traj_tensors:
            mask_val = 0.0 if es.step == 0 and es.done else 1.0
            self.traj_tensors["mask"][es.traj_idx, es.step] = mask_val

        msg = struct.pack(REQ_FMT, es.traj_idx, es.step,
                          self.worker_idx, es.env_idx, op)
        self.bus.send("infer_req", msg)
        es.pending = True

    # ------------------------------------------------------------------
    # Advance a single environment
    # ------------------------------------------------------------------

    def _advance_single(self, es: EnvState) -> None:
        ti = es.traj_idx
        s = es.step

        # Read action from shared tensor (inference server already wrote it)
        action_t = self._act_tensor[ti, s]
        if self.act_discrete:
            action = int(action_t[0].item())
        else:
            action = action_t.numpy().copy()

        # env.step
        next_obs, reward, terminated, truncated, info = es.env.step(action)
        done = bool(terminated or truncated)

        # Write reward / done into shared tensors
        self._rew_tensor[ti, s] = reward
        self._done_tensor[ti, s] = int(done)
        if "terminated" in self.traj_tensors:
            self.traj_tensors["terminated"][ti, s] = int(terminated)
        if "truncated" in self.traj_tensors:
            self.traj_tensors["truncated"][ti, s] = int(truncated)
        if "next_obs" in self.traj_tensors:
            self.traj_tensors["next_obs"][ti, s] = torch.from_numpy(
                np.asarray(next_obs, dtype=np.float32))

        # Episode stats
        es.episode_reward += reward
        es.episode_length += 1
        es.done = done

        if done:
            if self.worker_idx == 0 and info and "episode" in info:
                step = int(self.ctx.global_step.value)
                log_scalar(run="actor", tag="episode_reward", value=info["episode"]["r"], step=step)
                log_scalar(run="actor", tag="episode_length", value=info["episode"]["l"], step=step)

            next_obs, _ = es.env.reset()
            es.episode_reward = 0.0
            es.episode_length = 0

            # Zero LSTM state on reset
            if self.use_lstm:
                next_step = s + 1
                if next_step < self.traj_tensors["rnn_state_h"].shape[1]:
                    self.traj_tensors["rnn_state_h"][ti, next_step] = 0.0
                    self.traj_tensors["rnn_state_c"][ti, next_step] = 0.0

        es.obs = next_obs
        es.step += 1
        es.pending = False

        # Trajectory complete -> finalize, publish, get new buffer
        if es.step >= self.T:
            self._finalize_trajectory(es)

    # ------------------------------------------------------------------
    # Trajectory finalization
    # ------------------------------------------------------------------

    def _finalize_trajectory(self, es: EnvState) -> None:
        # Optional bootstrap: write obs[T] and request VALUE
        if self.bootstrap_value and not es.done:
            self._obs_tensor[es.traj_idx, self.T] = torch.from_numpy(
                np.asarray(es.obs, dtype=np.float32))
            # Send value request and busy-wait for flag
            self._send_request_value(es)
            while not self.ready_flags[self.worker_idx, es.env_idx]:
                pass  # spin — value requests are rare and fast
            self.ready_flags[self.worker_idx, es.env_idx] = 0
        elif "value" in self.traj_tensors:
            self.traj_tensors["value"][es.traj_idx, self.T] = 0.0

        # Publish filled trajectory
        self.bus.send("filled_out", struct.pack("<i", es.traj_idx))

        # Update global step counter
        with self.ctx.global_step.get_lock():
            self.ctx.global_step.value += self.T

        if self.worker_idx == 0:
            step = int(self.ctx.global_step.value)
            elapsed = time.time() - self.ctx.start_time
            log_scalar(run="actor", tag="steps", value=step, step=step)
            if elapsed > 0:
                log_scalar(run="actor", tag="fps", value=step / elapsed, step=step)

        # Get new trajectory buffer
        try:
            es.traj_idx = self.traj_queue.get(timeout=10.0)
        except Empty:
            logging.error("[worker:%d] timeout waiting for free traj buffer", self.worker_idx)
            return

        es.step = 0
        es.done = False

        # Initialize LSTM state for new trajectory
        if self.use_lstm:
            self.traj_tensors["rnn_state_h"][es.traj_idx, 0] = 0.0
            self.traj_tensors["rnn_state_c"][es.traj_idx, 0] = 0.0

    def _send_request_value(self, es: EnvState) -> None:
        """Send a VALUE-only request (bootstrap at trajectory boundary)."""
        msg = struct.pack(REQ_FMT, es.traj_idx, self.T,
                          self.worker_idx, es.env_idx, OP_VALUE)
        self.bus.send("infer_req", msg)
        es.pending = True


# ------------------------------------------------------------------
# Process entry point
# ------------------------------------------------------------------

def main(ctx, worker_idx: int, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[worker:%d] starting", worker_idx)
    worker = RolloutWorker(ctx, worker_idx)
    worker.run()
    logging.info("[worker:%d] stopped", worker_idx)
