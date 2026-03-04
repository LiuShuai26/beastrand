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

from nodes.common import child_logging_setup, child_sig_setup, ProfileAccum
from nodes.logger import child_attach_logger, log_scalar
from strandbus.strandbus import StrandBus
from utils.import_utils import get_object_from_path
from utils.model_sharing import ParameterClient

# Message format: traj_idx (i), step (i), worker_idx (i), env_idx (i), op (i)
REQ_FMT = "<iiiii"
REQ_SIZE = struct.calcsize(REQ_FMT)

OP_ACT = 0
OP_VALUE = 1


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
        base = ctx.ipc_dir
        self.bus.open("req", mode="pull", endpoint=f"{base}/infer.req", bind=True)

        # -- Profiling --
        self.prof = ProfileAccum(interval=5.0)

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

    def _gather_obs(self, traj_idxs: np.ndarray, steps: np.ndarray) -> Dict[str, Any]:
        """Gather obs (+ LSTM states) from shared tensors.

        Uses scalar indexing + torch.stack instead of advanced tensor indexing.
        For small batches on CPU shared-memory tensors, this is ~20x faster
        because it avoids the generic gather kernel overhead.
        """
        obs_t = self.traj_tensors["obs"]
        obs_batch = torch.stack(
            [obs_t[ti, s] for ti, s in zip(traj_idxs, steps)], dim=0
        )
        if self.device.type != "cpu":
            obs_batch = obs_batch.to(self.device)

        inputs: Dict[str, Any] = {"obs": obs_batch}

        if self.use_lstm and "rnn_state_h" in self.traj_tensors:
            h_t = self.traj_tensors["rnn_state_h"]
            c_t = self.traj_tensors["rnn_state_c"]
            h = torch.stack(
                [h_t[ti, s] for ti, s in zip(traj_idxs, steps)], dim=0
            ).unsqueeze(0)  # (1, N, hidden)
            c = torch.stack(
                [c_t[ti, s] for ti, s in zip(traj_idxs, steps)], dim=0
            ).unsqueeze(0)
            if self.device.type != "cpu":
                h, c = h.to(self.device), c.to(self.device)
            inputs["rnn_state"] = (h, c)
            if "mask" in self.traj_tensors:
                m_t = self.traj_tensors["mask"]
                m = torch.stack(
                    [m_t[ti, s] for ti, s in zip(traj_idxs, steps)], dim=0
                )
                if self.device.type != "cpu":
                    m = m.to(self.device)
                inputs["mask"] = m

        return inputs

    def _set_ready_flags(self, worker_ids: np.ndarray, env_idxs: np.ndarray) -> None:
        """Vectorized ready flag setting."""
        wi_t = torch.from_numpy(worker_ids.astype(np.int64))
        ei_t = torch.from_numpy(env_idxs.astype(np.int64))
        self.ready_flags[wi_t, ei_t] = 1

    def _run_act(self, reqs: np.ndarray) -> None:
        """Gather obs -> forward -> scatter action/logp/value -> set ready flags.

        reqs: Nx5 int32 array [traj_idx, step, worker_idx, env_idx, op]
        """
        t_start = time.monotonic()

        traj_idxs = reqs[:, 0]
        steps = reqs[:, 1]
        worker_ids = reqs[:, 2]
        env_idxs = reqs[:, 3]

        # Gather obs (scalar indexing + stack, fast for small batches on shared memory)
        inputs = self._gather_obs(traj_idxs, steps)
        t_gather = time.monotonic()
        self.prof.add("gather_obs", t_gather - t_start)

        # Forward pass
        out = self.policy.act(inputs, deterministic=False)
        t_fwd = time.monotonic()
        self.prof.add("forward", t_fwd - t_gather)

        # --- Scatter results back: advanced indexing (write path, less overhead) ---
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

        # Set ready flags (vectorized)
        self._set_ready_flags(worker_ids, env_idxs)

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

        inputs = self._gather_obs(traj_idxs, steps)

        v = self.policy.value(inputs)
        values = v if self.device.type == "cpu" else v.cpu()

        ti_t = torch.from_numpy(traj_idxs.astype(np.int64))
        s_t = torch.from_numpy(steps.astype(np.int64))
        self.traj_tensors["value"][ti_t, s_t] = values

        self._set_ready_flags(worker_ids, env_idxs)


def main(ctx, logger_queue) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[inference] starting")
    InferenceServer(ctx).serve()
    logging.info("[inference] stopped")
