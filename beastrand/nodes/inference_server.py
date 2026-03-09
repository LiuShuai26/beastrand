"""
InferenceServer (v4): vectorized I/O via compact inference buffers.

Data flow:
  1. Worker writes obs into infer_obs[worker_idx, env_idx] (compact, L2-resident)
  2. Worker sends lightweight request: struct.pack("<iii", worker_idx, env_idx, op)  (12 bytes)
  3. InferenceServer gathers obs via vectorized infer_obs[wi, ei] (no Python loop)
  4. InferenceServer scatters action/logp/value into infer_act/logp/val[wi, ei] (vectorized)
  5. InferenceServer sets ready_flags[worker_idx, env_idx] = 1
  6. Worker reads results from infer_* buffers and writes to traj_tensors for training

No pickle on the hot path. Compact [W, E, *shape] inference buffers replace scattered
[num_traj, T+1, *shape] traj_tensor access on the IS hot path.
"""
from __future__ import annotations

import logging
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from beastrand.nodes.common import child_logging_setup, child_sig_setup, ProfileAccum
from beastrand.nodes.logger import child_attach_logger, log_scalar
from beastrand.strandbus.strandbus import StrandBus
from beastrand.utils.import_utils import get_object_from_path
from beastrand.utils.model_sharing import ParameterClient

# Message format: worker_idx (i), env_idx (i), op (i)  — 12 bytes
REQ_FMT = "<iii"
REQ_SIZE = struct.calcsize(REQ_FMT)

OP_ACT = 0
OP_VALUE = 1


class InferenceServer:
    def __init__(self, ctx, server_idx: int = 0):
        self.ctx = ctx
        self.server_idx = server_idx
        self.device = torch.device(getattr(ctx.args, "inference_device", "cpu"))

        # -- policy --
        policy_cls = get_object_from_path(ctx.args.policy_path)
        self.policy = policy_cls(ctx).to(self.device)
        self.policy.eval()
        self.use_lstm = bool(getattr(self.policy, "use_lstm", False))

        # -- shared tensors (set by serve()) --
        self.traj_tensors: Dict[str, torch.Tensor] = {}
        self.ready_flags: Optional[torch.Tensor] = None
        self.param_client: Optional[ParameterClient] = None
        # Compact live rnn_state buffers [num_workers, num_envs, hidden] (LSTM only).
        self.rnn_state_live_h: Optional[torch.Tensor] = None
        self.rnn_state_live_c: Optional[torch.Tensor] = None
        # Compact inference I/O buffers [num_workers, num_envs, *shape].
        self.infer_obs: Optional[torch.Tensor] = None
        self.infer_act: Optional[torch.Tensor] = None
        self.infer_logp: Optional[torch.Tensor] = None
        self.infer_val: Optional[torch.Tensor] = None
        self.infer_mask: Optional[torch.Tensor] = None
        self.infer_action_logits: Optional[torch.Tensor] = None

        # -- ZMQ (only for receiving requests, no reply sockets) --
        self.bus = StrandBus()
        base = ctx.ipc_dir
        self.bus.open("req", mode="pull", endpoint=f"{base}/infer_{server_idx}.req", bind=True)

        # -- Profiling --
        self.prof = ProfileAccum(interval=5.0)

    # ------------------------------------------------------------------
    # Core inference loop
    # ------------------------------------------------------------------

    def serve(self):
        # Attach shared resources from ctx (set by Manager before spawn)
        self.traj_tensors = self.ctx.buffer_mgr.traj_tensors
        self.ready_flags = self.ctx.buffer_mgr.ready_flags
        self.rnn_state_live_h = self.ctx.buffer_mgr.rnn_state_live_h
        self.rnn_state_live_c = self.ctx.buffer_mgr.rnn_state_live_c
        self.infer_obs = self.ctx.buffer_mgr.infer_obs
        self.infer_act = self.ctx.buffer_mgr.infer_act
        self.infer_logp = self.ctx.buffer_mgr.infer_logp
        self.infer_val = self.ctx.buffer_mgr.infer_val
        self.infer_mask = self.ctx.buffer_mgr.infer_mask
        self.infer_action_logits = self.ctx.buffer_mgr.infer_action_logits
        self.param_client = ParameterClient(self.ctx.param_server)

        # Initial weight load
        self.param_client.ensure_updated(self.policy)
        logging.info("[inference:%d] ready (device=%s, lstm=%s)", self.server_idx, self.device, self.use_lstm)

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
        """Parse raw ZMQ messages into Nx3 int32 array: [worker_idx, env_idx, op]."""
        buf = b"".join(raw_msgs)
        total_bytes = len(buf)
        if total_bytes == 0 or total_bytes % REQ_SIZE != 0:
            return np.empty((0, 3), dtype=np.int32)
        n = total_bytes // REQ_SIZE
        return np.frombuffer(buf, dtype=np.int32).reshape(n, 3)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _process_batch(self, requests: np.ndarray) -> None:
        ops = requests[:, 2]
        act_mask = ops == OP_ACT
        val_mask = ops == OP_VALUE

        if act_mask.any():
            self._run_act(requests[act_mask])
        if val_mask.any():
            self._run_value(requests[val_mask])

    def _gather_obs(
        self,
        worker_idxs: np.ndarray,
        env_idxs: np.ndarray,
    ):
        """Vectorized gather from compact inference I/O buffers.

        obs: read from infer_obs[worker_idx, env_idx] — single vectorized index,
          no Python loop, no random scatter into large traj_tensors.
        rnn_state: read from rnn_state_live[worker_idx, env_idx] — L2-resident.
        """
        wi = torch.from_numpy(worker_idxs.astype(np.int64))
        ei = torch.from_numpy(env_idxs.astype(np.int64))

        obs_batch = self.infer_obs[wi, ei]  # (N, *obs_shape), vectorized
        if self.device.type != "cpu":
            obs_batch = obs_batch.to(self.device)

        inputs: Dict[str, Any] = {"obs": obs_batch}

        if self.use_lstm:
            if self.rnn_state_live_h is not None:
                h = self.rnn_state_live_h[wi, ei].unsqueeze(0)  # (1, N, hidden)
                c = self.rnn_state_live_c[wi, ei].unsqueeze(0)
            else:
                h = torch.zeros(1, len(wi), self.policy.lstm_hidden_size)
                c = torch.zeros_like(h)
            if self.device.type != "cpu":
                h, c = h.to(self.device), c.to(self.device)
            inputs["rnn_state"] = (h, c)
            # mask: read from infer_mask[wi, ei]
            m = self.infer_mask[wi, ei]
            if self.device.type != "cpu":
                m = m.to(self.device)
            inputs["mask"] = m

        return inputs, wi, ei

    def _set_ready_flags(self, wi: torch.Tensor, ei: torch.Tensor) -> None:
        """Vectorized ready flag setting."""
        self.ready_flags[wi, ei] = 1

    def _run_act(self, reqs: np.ndarray) -> None:
        """Gather obs from infer_obs -> forward -> scatter to infer_act/logp/val -> set flags.

        reqs: Nx3 int32 array [worker_idx, env_idx, op]
        """
        t_start = time.monotonic()

        worker_ids = reqs[:, 0]
        env_idxs = reqs[:, 1]

        # Vectorized gather from compact inference buffers (no Python loop)
        inputs, wi, ei = self._gather_obs(worker_ids, env_idxs)
        t_gather = time.monotonic()
        self.prof.add("gather_obs", t_gather - t_start)

        # Forward pass
        out = self.policy.act(inputs, deterministic=False)
        t_fwd = time.monotonic()
        self.prof.add("forward", t_fwd - t_gather)

        # --- Scatter results into compact infer_* buffers (vectorized) ---
        actions = out["action"] if self.device.type == "cpu" else out["action"].cpu()
        self.infer_act[wi, ei] = actions

        if "logp" in out:
            logps = out["logp"] if self.device.type == "cpu" else out["logp"].cpu()
            self.infer_logp[wi, ei] = logps

        if "value" in out:
            values = out["value"] if self.device.type == "cpu" else out["value"].cpu()
            self.infer_val[wi, ei] = values

        if "action_logits" in out and self.infer_action_logits is not None:
            al = out["action_logits"] if self.device.type == "cpu" else out["action_logits"].cpu()
            self.infer_action_logits[wi, ei] = al

        if self.use_lstm and "rnn_state" in out:
            h_out, c_out = out["rnn_state"]
            if self.device.type != "cpu":
                h_out, c_out = h_out.cpu(), c_out.cpu()
            h_out = h_out.squeeze(0)  # (N, hidden)
            c_out = c_out.squeeze(0)
            # Write to compact live buffer (vectorized, for next inference gather).
            if self.rnn_state_live_h is not None:
                self.rnn_state_live_h[wi, ei] = h_out
                self.rnn_state_live_c[wi, ei] = c_out

        t_scatter = time.monotonic()
        self.prof.add("scatter", t_scatter - t_fwd)

        # Set ready flags (vectorized)
        self._set_ready_flags(wi, ei)

        t_signal = time.monotonic()
        self.prof.add("set_flags", t_signal - t_scatter)

    def _run_value(self, reqs: np.ndarray) -> None:
        """VALUE-only requests (bootstrap at trajectory boundary).

        reqs: Nx3 int32 array [worker_idx, env_idx, op]
        """
        worker_ids = reqs[:, 0]
        env_idxs = reqs[:, 1]

        inputs, wi, ei = self._gather_obs(worker_ids, env_idxs)

        v = self.policy.value(inputs)
        values = v if self.device.type == "cpu" else v.cpu()
        self.infer_val[wi, ei] = values

        self._set_ready_flags(wi, ei)


def main(ctx, logger_queue, server_idx: int = 0) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[inference:%d] starting", server_idx)
    InferenceServer(ctx, server_idx=server_idx).serve()
    logging.info("[inference:%d] stopped", server_idx)
