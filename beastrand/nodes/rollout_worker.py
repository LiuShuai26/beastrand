"""
RolloutWorker (v4): async per-env rollout via shared memory flags + compact I/O buffers.

Each worker manages ``num_envs_per_worker`` environments. Every env
independently cycles through: send inference request → poll ready_flag →
step → send next request. This per-env async design naturally pipelines
CPU env.step() with GPU inference — no explicit double-buffering needed.

Workers write obs to infer_obs[worker_idx, env_idx] before each request.
After the IS sets ready_flags, workers read action/logp/value from the
compact infer_* buffers (vectorized [W, E] layout) and write them to
traj_tensors for training storage. rnn_state is written to traj_tensors
by the worker (not the IS), keeping traj_tensors writes on the worker side.
"""
from __future__ import annotations

import logging
import struct
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List

import numpy as np
import torch

from beastrand.core.envs.make_env import make_env
from beastrand.nodes.common import child_logging_setup, child_sig_setup, ProfileAccum
from beastrand.nodes.logger import child_attach_logger, log_scalar
from beastrand.strandbus.strandbus import StrandBus

# Must match inference_server.py
REQ_FMT = "<iii"
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


class RolloutWorker:
    def __init__(self, ctx, worker_idx: int):
        self.ctx = ctx
        self.worker_idx = worker_idx
        args = ctx.args

        self.T = args.rollout
        self.num_envs = getattr(args, "num_envs_per_worker", 2)

        self.use_lstm = bool(getattr(args, "use_lstm", False))
        self.bootstrap_value = bool(args.bootstrap_value)

        # Shared tensors from BufferMgr
        self.traj_tensors = ctx.buffer_mgr.traj_tensors
        self.traj_queue = ctx.buffer_mgr.traj_buffer_queue
        self.ready_flags = ctx.buffer_mgr.ready_flags

        # Action space info (set by Manager on ctx)
        self.act_discrete = ctx.act_kind == "discrete"

        # Pre-cache frequently used traj_tensors (training storage)
        self._rew_tensor = self.traj_tensors["reward"]
        self._done_tensor = self.traj_tensors["done"]
        # Live rnn_state buffers [num_workers, num_envs, hidden] (None for non-LSTM).
        self._rnn_live_h = ctx.buffer_mgr.rnn_state_live_h
        self._rnn_live_c = ctx.buffer_mgr.rnn_state_live_c
        # Compact inference I/O buffers (shared with IS, indexed [worker, env])
        self._infer_obs = ctx.buffer_mgr.infer_obs
        self._infer_act = ctx.buffer_mgr.infer_act
        self._infer_logp = ctx.buffer_mgr.infer_logp
        self._infer_val = ctx.buffer_mgr.infer_val
        self._infer_mask = ctx.buffer_mgr.infer_mask
        self._infer_action_logits = ctx.buffer_mgr.infer_action_logits

        # --- ZMQ (only for sending requests + filled trajectories) ---
        self.bus = StrandBus()
        base = ctx.ipc_dir
        num_infer = getattr(args, "num_inference_servers", 1)
        self.bus.open("infer_req", mode="push", endpoint=f"{base}/infer_0.req", bind=False)
        for i in range(1, num_infer):
            self.bus.sockets["infer_req"].connect(f"{base}/infer_{i}.req")
        self.bus.open("filled_out", mode="push", endpoint=f"{base}/data.filled.in", bind=False)

        # --- Profiling & stats (only worker 0) ---
        self.prof = ProfileAccum(interval=5.0) if worker_idx == 0 else None
        self._recent_rewards: deque = deque(maxlen=100) if worker_idx == 0 else None
        self._recent_lengths: deque = deque(maxlen=100) if worker_idx == 0 else None
        self._last_summary_time: float = 0.0

        # --- Resolve env factory (configurable via args.make_env_path) ---
        _make_env_path = getattr(args, "make_env_path", None)
        if _make_env_path:
            from beastrand.utils.import_utils import get_object_from_path
            _make_env = get_object_from_path(_make_env_path)
        else:
            _make_env = make_env

        # --- Create all environments ---
        self.envs: List[EnvState] = []
        for ei in range(self.num_envs):
            seed = args.seed + worker_idx * self.num_envs + ei
            env = _make_env(args.env_id, seed=seed, args=args)
            obs, _ = env.reset(seed=seed)
            traj_idx = self.traj_queue.get()
            self.envs.append(EnvState(
                env=env, obs=obs, traj_idx=traj_idx, env_idx=ei,
            ))


    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self):
        logging.info("[worker:%d] ready (%d envs, T=%d)",
                     self.worker_idx, self.num_envs, self.T)

        # Initial: send inference requests for all envs
        for es in self.envs:
            self._send_request(es, OP_ACT)

        try:
            self._loop_per_env()
        finally:
            self.bus.close_all()
            logging.info("[worker:%d] exiting", self.worker_idx)

    def _loop_per_env(self):
        """Original per-env loop: advance → send immediately."""
        while not self.ctx.stop_event.is_set():
            t0 = time.monotonic()
            stepped_any = False

            for es in self.envs:
                if not es.pending:
                    continue
                if not self.ready_flags[self.worker_idx, es.env_idx]:
                    continue

                self.ready_flags[self.worker_idx, es.env_idx] = 0
                t1 = time.monotonic()
                self._advance_single(es)
                t2 = time.monotonic()

                if self.prof:
                    self.prof.add("advance", t2 - t1)

                if es.step < self.T:
                    self._send_request(es, OP_ACT)
                stepped_any = True

            if self.prof:
                t_end = time.monotonic()
                if not stepped_any:
                    self.prof.add("idle_spin", t_end - t0)
                report = self.prof.maybe_report(f"worker:{self.worker_idx}")
                if report:
                    logging.info(report)

    # ------------------------------------------------------------------
    # Send inference request (single env)
    # ------------------------------------------------------------------

    def _send_request(self, es: EnvState, op: int) -> None:
        wi = self.worker_idx
        ei = es.env_idx
        ti = es.traj_idx
        s = es.step

        obs_t = torch.from_numpy(np.asarray(es.obs, dtype=np.float32))

        # Write obs to compact infer_obs (for IS vectorized gather)
        self._infer_obs[wi, ei] = obs_t
        # Also write to traj_tensors["obs"] for training storage
        self.traj_tensors["obs"][ti, s] = obs_t

        if self.use_lstm:
            # Capture INPUT rnn_state before IS overwrites it with the output state
            if self._rnn_live_h is not None and "rnn_state_h" in self.traj_tensors:
                self.traj_tensors["rnn_state_h"][ti, s] = self._rnn_live_h[wi, ei]
                self.traj_tensors["rnn_state_c"][ti, s] = self._rnn_live_c[wi, ei]
            # Write mask (IS reads this for the forward pass)
            mask_val = 0.0 if es.done else 1.0
            self._infer_mask[wi, ei] = mask_val
            if "mask" in self.traj_tensors:
                self.traj_tensors["mask"][ti, s] = mask_val

        msg = struct.pack(REQ_FMT, wi, ei, op)
        self.bus.send("infer_req", msg)
        es.pending = True

    # ------------------------------------------------------------------
    # Advance a single environment
    # ------------------------------------------------------------------

    def _advance_single(self, es: EnvState) -> None:
        ti = es.traj_idx
        s = es.step
        wi = self.worker_idx
        ei = es.env_idx

        # Read action/logp/value from compact infer_* buffers, write to traj_tensors.
        # obs/mask/rnn_state were already written to traj_tensors in _send_request.
        act_val = self._infer_act[wi, ei]
        if self.act_discrete:
            action = int(act_val[0].item())
        else:
            action = act_val.numpy().copy()
        self.traj_tensors["action"][ti, s] = act_val
        if "log_prob" in self.traj_tensors:
            self.traj_tensors["log_prob"][ti, s] = self._infer_logp[wi, ei]
        if "value" in self.traj_tensors:
            self.traj_tensors["value"][ti, s] = self._infer_val[wi, ei]
        if self._infer_action_logits is not None and "action_logits" in self.traj_tensors:
            self.traj_tensors["action_logits"][ti, s] = self._infer_action_logits[wi, ei]
        if "model_version" in self.traj_tensors:
            self.traj_tensors["model_version"][ti, s] = int(self.ctx.buffer_mgr.policy_version.item())

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
            if self.worker_idx == 0:
                step = int(self.ctx.global_step.value)
                # Use raw reward from RecordEpisodeStatistics when available
                # (before NormalizeReward); fall back to accumulated reward
                # (already raw for envs without normalization wrappers)
                raw_reward = es.episode_reward
                if info and "episode" in info:
                    raw_reward = float(info["episode"]["r"])
                self._recent_rewards.append(raw_reward)
                self._recent_lengths.append(es.episode_length)
                log_scalar(run="actor", tag="episode_reward", value=raw_reward, step=step)
                log_scalar(run="actor", tag="episode_length", value=es.episode_length, step=step)
                self._maybe_log_summary(step)

            next_obs, _ = es.env.reset()
            es.episode_reward = 0.0
            es.episode_length = 0

            # Zero LSTM live state on episode reset (fresh hidden state for next step).
            # traj_tensors["rnn_state_h"][ti, s+1] will be written in _send_request
            # when the next step is submitted, reading from this zeroed live buffer.
            if self.use_lstm and self._rnn_live_h is not None:
                self._rnn_live_h[self.worker_idx, es.env_idx] = 0.0
                self._rnn_live_c[self.worker_idx, es.env_idx] = 0.0

        es.obs = next_obs
        es.step += 1
        es.pending = False

        # Trajectory complete -> finalize, publish, get new buffer
        if es.step >= self.T:
            self._finalize_trajectory(es)

    def _maybe_log_summary(self, step: int) -> None:
        """Print aggregated episode stats periodically (worker 0 only)."""
        now = time.time()
        if now - self._last_summary_time < 5.0:
            return
        if not self._recent_rewards:
            return
        n = len(self._recent_rewards)
        avg_r = sum(self._recent_rewards) / n
        avg_l = sum(self._recent_lengths) / n
        logging.info("avg_reward=%.2f avg_length=%.0f episodes=%d steps=%d",
                     avg_r, avg_l, n, step)
        log_scalar(run="actor", tag="avg_reward", value=avg_r, step=step)
        self._last_summary_time = now

    # ------------------------------------------------------------------
    # Trajectory finalization
    # ------------------------------------------------------------------

    def _finalize_trajectory(self, es: EnvState) -> None:
        # Optional bootstrap: request VALUE for obs[T] (next obs after last step)
        if self.bootstrap_value and not es.done:
            wi = self.worker_idx
            ei = es.env_idx
            # Write bootstrap obs to infer_obs (IS reads from here)
            self._infer_obs[wi, ei] = torch.from_numpy(
                np.asarray(es.obs, dtype=np.float32))
            # Send value request and busy-wait for flag
            self._send_request_value(es)
            while not self.ready_flags[wi, ei]:
                pass  # spin — value requests are rare and fast
            self.ready_flags[wi, ei] = 0
            # Write bootstrap value to traj_tensors
            if "value" in self.traj_tensors:
                self.traj_tensors["value"][es.traj_idx, self.T] = self._infer_val[wi, ei]
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

        # Get new trajectory buffer.  Use short timeouts in a loop so we
        # can exit promptly when stop_event fires instead of blocking for
        # the full 10 seconds.
        got_buffer = False
        for _ in range(20):  # 20 × 0.5s = 10s total
            if self.ctx.stop_event.is_set():
                return
            try:
                es.traj_idx = self.traj_queue.get(timeout=0.5)
                got_buffer = True
                break
            except Empty:
                continue

        if not got_buffer:
            logging.error("[worker:%d] timeout waiting for free traj buffer", self.worker_idx)
            return

        es.step = 0
        es.done = False

        # Reset LSTM live state at trajectory boundary (always fresh hidden state).
        # traj_tensors["rnn_state_h"][new_ti, 0] will be written in _send_request.
        if self.use_lstm and self._rnn_live_h is not None:
            self._rnn_live_h[self.worker_idx, es.env_idx] = 0.0
            self._rnn_live_c[self.worker_idx, es.env_idx] = 0.0

    def _send_request_value(self, es: EnvState) -> None:
        """Send a VALUE-only request (bootstrap at trajectory boundary)."""
        msg = struct.pack(REQ_FMT, self.worker_idx, es.env_idx, OP_VALUE)
        self.bus.send("infer_req", msg)
        es.pending = True


# ------------------------------------------------------------------
# Process entry point
# ------------------------------------------------------------------

def main(ctx, worker_idx: int, logger_queue) -> None:
    # NOTE: No CPU core pinning (os.sched_setaffinity / psutil.cpu_affinity).
    # The OS scheduler handles core assignment well enough for our env.step()
    # latency (~0.1-1ms). Consider adding affinity if scaling to 96+ vCPU
    # or running on NUMA machines where cross-node memory access hurts.
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[worker:%d] starting", worker_idx)
    worker = RolloutWorker(ctx, worker_idx)
    worker.run()
    logging.info("[worker:%d] stopped", worker_idx)
