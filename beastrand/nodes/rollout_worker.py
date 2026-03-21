"""
RolloutWorker (v5): fixed-allocation slots, flat inference I/O buffers.

Each (worker, env) pair owns split_depth trajectory slots. Slot assignment:
  flat_idx = worker_idx * num_envs_per_worker + env_idx
  traj_idx = flat_idx * split_depth + split_idx

Workers alternate split_idx each trajectory. The Learner releases the slot
via split_flags[flat_idx, split_idx] = 0; the worker spin-waits on this flag
before reusing the slot (replaces traj_buffer_queue.get()).

Inference I/O buffers are flat [W*E, *shape]. IS request message is 8 bytes:
  struct.pack("<ii", flat_idx, op)

Workers write obs to infer_obs[flat_idx] before each request. After the IS
sets ready_flags[flat_idx], workers read action/logp/value from the compact
infer_* buffers and write them to traj_tensors for training storage.
"""
from __future__ import annotations

import logging
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from beastrand.core.envs.make_env import make_env
from beastrand.nodes.common import child_logging_setup, child_sig_setup, ProfileAccum
from beastrand.nodes.logger import child_attach_logger, log_scalar
from beastrand.strandbus.strandbus import StrandBus

# Must match inference_server.py
REQ_FMT = "<ii"
REQ_SIZE = struct.calcsize(REQ_FMT)  # = 8 bytes
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
        self.split_depth = ctx.buffer_mgr.split_depth

        self.use_lstm = bool(getattr(args, "use_lstm", False))
        self.bootstrap_value = bool(args.bootstrap_value)

        # Shared tensors from BufferMgr
        self.traj_tensors = ctx.buffer_mgr.traj_tensors
        self.ready_flags = ctx.buffer_mgr.ready_flags       # [W*E] flat
        self.split_flags = ctx.buffer_mgr.split_flags       # [W*E, split_depth]

        # Action space info (set by Manager on ctx)
        self.act_discrete = ctx.act_kind == "discrete"

        # Pre-cache frequently used traj_tensors (training storage)
        self._rew_tensor = self.traj_tensors["reward"]
        self._done_tensor = self.traj_tensors["done"]
        # Live rnn_state buffers [W*E, hidden] flat (None for non-LSTM).
        self._rnn_live_h = ctx.buffer_mgr.rnn_state_live_h
        self._rnn_live_c = ctx.buffer_mgr.rnn_state_live_c
        # Compact inference I/O buffers (shared with IS, flat [W*E, *shape])
        self._infer_obs = ctx.buffer_mgr.infer_obs
        self._infer_act = ctx.buffer_mgr.infer_act
        self._infer_logp = ctx.buffer_mgr.infer_logp
        self._infer_val = ctx.buffer_mgr.infer_val
        self._infer_mask = ctx.buffer_mgr.infer_mask
        self._infer_action_logits = ctx.buffer_mgr.infer_action_logits

        # Per-env split index (local, alternates 0/1/... each trajectory).
        self._env_split = np.zeros(self.num_envs, dtype=np.int32)

        # --- ZMQ (only for sending requests + filled trajectories) ---
        self.bus = StrandBus()
        base = ctx.ipc_dir
        num_infer = getattr(args, "num_inference_servers", 1)
        self.bus.open("infer_req", mode="push", endpoint=f"{base}/infer_0.req", bind=False)
        for i in range(1, num_infer):
            self.bus.sockets["infer_req"].connect(f"{base}/infer_{i}.req")
        self.bus.open("filled_out", mode="push", endpoint=f"{base}/data.filled.in", bind=False)

        # --- Profiling (only worker 0) & episode stats (all workers) ---
        self.prof = ProfileAccum(interval=5.0) if worker_idx == 0 else None
        self._recent_rewards: deque = deque(maxlen=100)
        self._recent_lengths: deque = deque(maxlen=100)
        self._last_summary_time: float = 0.0
        # Recent FPS window (worker 0 only)
        self._fps_last_step: int = 0
        self._fps_last_time: float = 0.0

        # --- Resolve env factory (configurable via args.make_env_path) ---
        _make_env_path = getattr(args, "make_env_path", None)
        if _make_env_path:
            from beastrand.utils.import_utils import get_object_from_path
            _make_env = get_object_from_path(_make_env_path)
        else:
            _make_env = make_env

        # --- Create all environments with fixed traj slot assignment ---
        self.envs: List[EnvState] = []
        wi = self.worker_idx
        for ei in range(self.num_envs):
            seed = args.seed + wi * self.num_envs + ei
            env = _make_env(args.env_id, seed=seed, args=args)
            obs, _ = env.reset(seed=seed)
            flat_idx = wi * self.num_envs + ei
            traj_idx = flat_idx * self.split_depth + 0  # start with split 0
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
        """Per-env async loop: advance → send immediately."""
        while not self.ctx.stop_event.is_set():
            t0 = time.monotonic()
            stepped_any = False

            for es in self.envs:
                if not es.pending:
                    continue
                flat_idx = self.worker_idx * self.num_envs + es.env_idx
                if not self.ready_flags[flat_idx]:
                    continue

                self.ready_flags[flat_idx] = 0
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
        flat_idx = wi * self.num_envs + ei
        ti = es.traj_idx
        s = es.step

        obs_t = torch.from_numpy(np.asarray(es.obs, dtype=np.float32))

        # Write obs to compact infer_obs[flat_idx] (for IS numpy gather)
        self._infer_obs[flat_idx] = obs_t
        # Also write to traj_tensors["obs"] for training storage
        self.traj_tensors["obs"][ti, s] = obs_t

        if self.use_lstm:
            # Capture INPUT rnn_state before IS overwrites it with the output state
            if self._rnn_live_h is not None and "rnn_state_h" in self.traj_tensors:
                self.traj_tensors["rnn_state_h"][ti, s] = self._rnn_live_h[flat_idx]
                self.traj_tensors["rnn_state_c"][ti, s] = self._rnn_live_c[flat_idx]
            # Write mask (IS reads this for the forward pass)
            mask_val = 0.0 if es.done else 1.0
            self._infer_mask[flat_idx] = mask_val
            if "mask" in self.traj_tensors:
                self.traj_tensors["mask"][ti, s] = mask_val

        msg = struct.pack(REQ_FMT, flat_idx, op)
        self.bus.send("infer_req", msg)
        es.pending = True

    # ------------------------------------------------------------------
    # Advance a single environment
    # ------------------------------------------------------------------

    def _advance_single(self, es: EnvState) -> None:
        ti = es.traj_idx
        s = es.step
        flat_idx = self.worker_idx * self.num_envs + es.env_idx

        # Read action/logp/value from compact infer_* buffers, write to traj_tensors.
        # obs/mask/rnn_state were already written to traj_tensors in _send_request.
        act_val = self._infer_act[flat_idx]
        if self.act_discrete:
            action = int(act_val[0].item())
        else:
            action = act_val.numpy().copy()
        self.traj_tensors["action"][ti, s] = act_val
        if "log_prob" in self.traj_tensors:
            self.traj_tensors["log_prob"][ti, s] = self._infer_logp[flat_idx]
        if "value" in self.traj_tensors:
            self.traj_tensors["value"][ti, s] = self._infer_val[flat_idx]
        if self._infer_action_logits is not None and "action_logits" in self.traj_tensors:
            self.traj_tensors["action_logits"][ti, s] = self._infer_action_logits[flat_idx]
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
            step = int(self.ctx.global_step.value)
            raw_reward = es.episode_reward
            if info and "episode" in info:
                raw_reward = float(info["episode"]["r"])
            self._recent_rewards.append(raw_reward)
            self._recent_lengths.append(es.episode_length)
            self._maybe_log_summary(step)

            next_obs, _ = es.env.reset()
            es.episode_reward = 0.0
            es.episode_length = 0

            # Zero LSTM live state on episode reset.
            if self.use_lstm and self._rnn_live_h is not None:
                self._rnn_live_h[flat_idx] = 0.0
                self._rnn_live_c[flat_idx] = 0.0

        es.obs = next_obs
        es.step += 1
        es.pending = False

        # Trajectory complete -> finalize, publish, advance split
        if es.step >= self.T:
            self._finalize_trajectory(es)

    def _maybe_log_summary(self, step: int) -> None:
        """Print aggregated episode stats periodically. Only worker 0 writes to TensorBoard."""
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
        if self.worker_idx == 0:
            log_scalar(run="actor", tag="avg_reward", value=avg_r, step=step)
            log_scalar(run="actor", tag="avg_ep_length", value=avg_l, step=step)
        self._last_summary_time = now

    # ------------------------------------------------------------------
    # Trajectory finalization
    # ------------------------------------------------------------------

    def _finalize_trajectory(self, es: EnvState) -> None:
        wi = self.worker_idx
        ei = es.env_idx
        flat_idx = wi * self.num_envs + ei
        split_idx = int(self._env_split[ei])

        # Optional bootstrap: request VALUE for obs[T] (next obs after last step)
        if self.bootstrap_value and not es.done:
            self._infer_obs[flat_idx] = torch.from_numpy(
                np.asarray(es.obs, dtype=np.float32))
            self._send_request_value(es)
            while not self.ready_flags[flat_idx]:
                if self.ctx.stop_event.is_set():
                    return
                time.sleep(0.001)
            self.ready_flags[flat_idx] = 0
            if "value" in self.traj_tensors:
                self.traj_tensors["value"][es.traj_idx, self.T] = self._infer_val[flat_idx]
        elif "value" in self.traj_tensors:
            self.traj_tensors["value"][es.traj_idx, self.T] = 0.0

        # Mark slot as learner-in-use BEFORE sending to DataServer
        self.split_flags[flat_idx, split_idx] = 1

        # Publish filled trajectory
        self.bus.send("filled_out", struct.pack("<i", es.traj_idx))

        # Update global step counter
        with self.ctx.global_step.get_lock():
            self.ctx.global_step.value += self.T

        if self.worker_idx == 0:
            step = int(self.ctx.global_step.value)
            now = time.time()
            elapsed = now - self.ctx.start_time
            log_scalar(run="actor", tag="steps", value=step, step=step)
            if elapsed > 0:
                log_scalar(run="actor", tag="fps", value=step / elapsed, step=step)
            # Recent FPS over the last window (more responsive than total average)
            if self._fps_last_time > 0:
                dt = now - self._fps_last_time
                if dt > 0:
                    recent_fps = (step - self._fps_last_step) / dt
                    log_scalar(run="actor", tag="fps_recent", value=recent_fps, step=step)
            self._fps_last_step = step
            self._fps_last_time = now

        # Advance to next split
        next_split = (split_idx + 1) % self.split_depth
        self._env_split[ei] = next_split

        # Spin-wait for the next split to be released by the Learner.
        # With split_depth=2, the worker can be exactly 1 trajectory ahead of the Learner.
        while not self.ctx.stop_event.is_set():
            if self.split_flags[flat_idx, next_split] == 0:
                break
            time.sleep(0.001)

        if self.ctx.stop_event.is_set():
            return

        # Assign new traj_idx for next trajectory
        es.traj_idx = flat_idx * self.split_depth + next_split
        es.step = 0
        es.done = False

        # Reset LSTM live state at trajectory boundary.
        if self.use_lstm and self._rnn_live_h is not None:
            self._rnn_live_h[flat_idx] = 0.0
            self._rnn_live_c[flat_idx] = 0.0

    def _send_request_value(self, es: EnvState) -> None:
        """Send a VALUE-only request (bootstrap at trajectory boundary)."""
        flat_idx = self.worker_idx * self.num_envs + es.env_idx
        # Sync mask so IS uses the correct LSTM reset boundary
        if self.use_lstm:
            self._infer_mask[flat_idx] = 0.0 if es.done else 1.0
        msg = struct.pack(REQ_FMT, flat_idx, OP_VALUE)
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
