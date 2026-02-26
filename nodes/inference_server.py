"""
InferenceServer (v3): zero-copy batched GPU inference with async signaling.

Data flow:
  1. Worker writes obs (+ rnn states) into traj_tensors[traj_idx, step]
  2. Worker sends lightweight request: struct.pack("<iiiii", traj_idx, step, worker_idx, env_idx, op)
  3. InferenceServer gathers obs from shared tensors, runs forward pass
  4. InferenceServer scatters action/logp/value back into shared tensors
  5. InferenceServer sets ready_flags[worker_idx, env_idx] = 1 (shared memory, no ZMQ reply)

No pickle on the hot path. All observation and action data stays in shared memory.
"""
from __future__ import annotations

import logging
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nodes.common import child_logging_setup, child_sig_setup
from nodes.logger import child_attach_logger, log_scalar
from strandbus.strandbus import StrandBus
from utils.import_utils import get_object_from_path
from utils.model_sharing import ParameterClient

# Message format: traj_idx (i), step (i), worker_idx (i), env_idx (i), op (i)
REQ_FMT = "<iiiii"
REQ_SIZE = struct.calcsize(REQ_FMT)

OP_ACT = 0
OP_VALUE = 1


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


class InferenceServer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.device = torch.device(getattr(ctx.args, "device", "cpu"))

        # -- policy --
        policy_cls = get_object_from_path(ctx.args.policy_path)
        self.policy = policy_cls(ctx).to(self.device)
        self.policy.eval()
        self.use_lstm = bool(getattr(self.policy, "use_lstm", False))

        # -- shared tensors (set by serve()) --
        self.traj_tensors: Dict[str, torch.Tensor] = {}
        self.ready_flags: Optional[torch.Tensor] = None
        self.param_client: Optional[ParameterClient] = None

        # -- ZMQ (only for receiving requests, no reply sockets) --
        self.bus = StrandBus()
        base = "ipc:///tmp/beatstrand"
        self.bus.open("req", mode="pull", endpoint=f"{base}/infer.req", bind=True)

        # -- Profiling --
        self.prof = _ProfileAccum(interval=5.0)

    # ------------------------------------------------------------------
    # Core inference loop
    # ------------------------------------------------------------------

    def serve(self):
        # Attach shared resources from ctx (set by Manager before spawn)
        self.traj_tensors = self.ctx.buffer_mgr.traj_tensors
        self.ready_flags = self.ctx.buffer_mgr.ready_flags
        self.param_client = ParameterClient(self.ctx.param_server)

        # Initial weight load
        self.param_client.ensure_updated(self.policy)
        logging.info("[inference] ready (device=%s, lstm=%s)", self.device, self.use_lstm)

        try:
            while not self.ctx.stop_event.is_set():
                # 1) Sync weights if learner has updated
                t0 = time.monotonic()
                self.param_client.ensure_updated(self.policy)
                t1 = time.monotonic()
                self.prof.add("weight_sync", t1 - t0)

                # 2) Receive batch of requests (block for first, drain rest)
                raw_msgs = self.bus.recv_many("req")
                t2 = time.monotonic()
                self.prof.add("recv", t2 - t1)

                # 3) Parse requests (batch via np.frombuffer)
                requests = self._parse_requests_fast(raw_msgs)
                t3 = time.monotonic()
                self.prof.add("parse", t3 - t2)
                self.prof.add("batch_size", len(requests))

                if len(requests) == 0:
                    continue

                # 4) Gather obs, run inference, scatter results, set flags
                self._process_batch(requests)

                # 5) Report profiling
                report = self.prof.maybe_report("inference")
                if report:
                    logging.info(report)

        finally:
            self.bus.close_all()
            logging.info("[inference] exiting")

    # ------------------------------------------------------------------
    # Request parsing
    # ------------------------------------------------------------------

    def _parse_requests_fast(self, raw_msgs: List[bytes]) -> np.ndarray:
        """Parse raw ZMQ messages into Nx5 int32 array:
        [traj_idx, step, worker_idx, env_idx, op].
        """
        buf = b"".join(raw_msgs)
        total_bytes = len(buf)
        if total_bytes == 0 or total_bytes % REQ_SIZE != 0:
            return np.empty((0, 5), dtype=np.int32)
        n = total_bytes // REQ_SIZE
        return np.frombuffer(buf, dtype=np.int32).reshape(n, 5)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _process_batch(self, requests: np.ndarray) -> None:
        ops = requests[:, 4]
        act_mask = ops == OP_ACT
        val_mask = ops == OP_VALUE

        if act_mask.any():
            self._run_act(requests[act_mask])
        if val_mask.any():
            self._run_value(requests[val_mask])

    def _run_act(self, reqs: np.ndarray) -> None:
        """Gather obs -> forward -> scatter action/logp/value -> set ready flags.

        reqs: Nx5 int32 array [traj_idx, step, worker_idx, env_idx, op]
        """
        t_start = time.monotonic()

        traj_idxs = reqs[:, 0]
        steps = reqs[:, 1]
        worker_ids = reqs[:, 2]
        env_idxs = reqs[:, 3]

        # Gather obs: list comp + stack (faster than advanced indexing for small batches)
        obs_tensor = self.traj_tensors["obs"]
        obs_slices = [obs_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
        obs_batch = torch.stack(obs_slices, dim=0)
        if self.device.type != "cpu":
            obs_batch = obs_batch.to(self.device)
        t_gather = time.monotonic()
        self.prof.add("gather_obs", t_gather - t_start)

        inputs: Dict[str, Any] = {"obs": obs_batch}

        # LSTM: gather rnn states and masks
        if self.use_lstm and "rnn_state_h" in self.traj_tensors:
            h_tensor = self.traj_tensors["rnn_state_h"]
            c_tensor = self.traj_tensors["rnn_state_c"]
            h_slices = [h_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
            c_slices = [c_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
            h = torch.stack(h_slices, dim=0).unsqueeze(0)  # (1, N, hidden)
            c = torch.stack(c_slices, dim=0).unsqueeze(0)
            if self.device.type != "cpu":
                h, c = h.to(self.device), c.to(self.device)
            inputs["rnn_state"] = (h, c)
            if "mask" in self.traj_tensors:
                m_tensor = self.traj_tensors["mask"]
                m_slices = [m_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
                m = torch.stack(m_slices, dim=0)
                if self.device.type != "cpu":
                    m = m.to(self.device)
                inputs["mask"] = m

        # Forward pass
        out = self.policy.act(inputs, deterministic=False)
        t_fwd = time.monotonic()
        self.prof.add("forward", t_fwd - t_gather)

        # --- Scatter results back: vectorized advanced indexing (no Python loop) ---
        ti_t = torch.from_numpy(traj_idxs.astype(np.int64))
        s_t = torch.from_numpy(steps.astype(np.int64))

        actions = out["action"] if self.device.type == "cpu" else out["action"].cpu()
        self.traj_tensors["action"][ti_t, s_t] = actions

        if "logp" in out:
            logps = out["logp"] if self.device.type == "cpu" else out["logp"].cpu()
            self.traj_tensors["log_prob"][ti_t, s_t] = logps

        if "value" in out:
            values = out["value"] if self.device.type == "cpu" else out["value"].cpu()
            self.traj_tensors["value"][ti_t, s_t] = values

        if self.use_lstm and "rnn_state" in out:
            h_out, c_out = out["rnn_state"]
            if self.device.type != "cpu":
                h_out, c_out = h_out.cpu(), c_out.cpu()
            h_out = h_out.squeeze(0)  # (N, hidden)
            c_out = c_out.squeeze(0)
            next_s = s_t + 1
            max_steps = self.traj_tensors["rnn_state_h"].shape[1]
            valid = next_s < max_steps
            if valid.any():
                vti = ti_t[valid]
                vns = next_s[valid]
                self.traj_tensors["rnn_state_h"][vti, vns] = h_out[valid]
                self.traj_tensors["rnn_state_c"][vti, vns] = c_out[valid]

        # Version stamp
        if "model_version" in self.traj_tensors:
            version = int(self.ctx.buffer_mgr.policy_version.item())
            self.traj_tensors["model_version"][ti_t, s_t] = version

        t_scatter = time.monotonic()
        self.prof.add("scatter", t_scatter - t_fwd)

        # Set ready flags (shared memory — workers poll these)
        ready = self.ready_flags
        for wid, eid in zip(worker_ids, env_idxs):
            ready[wid, eid] = 1

        t_signal = time.monotonic()
        self.prof.add("set_flags", t_signal - t_scatter)

    def _run_value(self, reqs: np.ndarray) -> None:
        """VALUE-only requests (bootstrap at trajectory boundary).

        reqs: Nx5 int32 array [traj_idx, step, worker_idx, env_idx, op]
        """
        traj_idxs = reqs[:, 0]
        steps = reqs[:, 1]
        worker_ids = reqs[:, 2]
        env_idxs = reqs[:, 3]

        obs_tensor = self.traj_tensors["obs"]
        obs_slices = [obs_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
        obs_batch = torch.stack(obs_slices, dim=0)
        if self.device.type != "cpu":
            obs_batch = obs_batch.to(self.device)

        inputs: Dict[str, Any] = {"obs": obs_batch}

        if self.use_lstm and "rnn_state_h" in self.traj_tensors:
            h_tensor = self.traj_tensors["rnn_state_h"]
            c_tensor = self.traj_tensors["rnn_state_c"]
            h_slices = [h_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
            c_slices = [c_tensor[ti, s] for ti, s in zip(traj_idxs, steps)]
            h = torch.stack(h_slices, dim=0).unsqueeze(0)
            c = torch.stack(c_slices, dim=0).unsqueeze(0)
            if self.device.type != "cpu":
                h, c = h.to(self.device), c.to(self.device)
            inputs["rnn_state"] = (h, c)

        v = self.policy.value(inputs)
        values = v if self.device.type == "cpu" else v.cpu()

        ti_t = torch.from_numpy(traj_idxs.astype(np.int64))
        s_t = torch.from_numpy(steps.astype(np.int64))
        self.traj_tensors["value"][ti_t, s_t] = values

        # Set ready flags
        ready = self.ready_flags
        for wid, eid in zip(worker_ids, env_idxs):
            ready[wid, eid] = 1


def main(ctx, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[inference] starting")
    InferenceServer(ctx).serve()
    logging.info("[inference] stopped")
