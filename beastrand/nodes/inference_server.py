"""
InferenceServer (v5): flat inference buffers, numpy 1D gather.

Data flow:
  1. Worker writes obs into infer_obs[flat_idx] (flat [W*E, *shape], L2-resident)
  2. Worker sends lightweight request: struct.pack("<ii", flat_idx, op)  (8 bytes)
  3. InferenceServer gathers obs via numpy 1D indexing (int32 native, no astype)
  4. InferenceServer scatters action/logp/value into infer_act/logp/val[flat_idx]
  5. InferenceServer sets ready_flags[flat_idx] = 1
  6. Worker reads results from infer_* buffers and writes to traj_tensors for training

flat_idx = worker_idx * num_envs_per_worker + env_idx

Numpy views are pre-cached in serve() so _gather_obs allocates nothing.
int32 flat indexing eliminates astype(int64) and PyTorch 2D fancy indexing.
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

# Message format: flat_idx (i), op (i)  — 8 bytes
REQ_FMT = "<ii"
REQ_SIZE = struct.calcsize(REQ_FMT)  # = 8

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
        # Compact live rnn_state buffers [W*E, hidden] flat (LSTM only).
        self.rnn_state_live_h: Optional[torch.Tensor] = None
        self.rnn_state_live_c: Optional[torch.Tensor] = None
        # Compact inference I/O buffers [W*E, *shape] flat.
        self.infer_obs: Optional[torch.Tensor] = None
        self.infer_act: Optional[torch.Tensor] = None
        self.infer_logp: Optional[torch.Tensor] = None
        self.infer_val: Optional[torch.Tensor] = None
        self.infer_mask: Optional[torch.Tensor] = None
        self.infer_action_logits: Optional[torch.Tensor] = None

        # Pre-cached numpy views (set by serve(), zero-copy into shared tensors)
        self._obs_np: Optional[np.ndarray] = None
        self._act_np: Optional[np.ndarray] = None
        self._logp_np: Optional[np.ndarray] = None
        self._val_np: Optional[np.ndarray] = None
        self._mask_np: Optional[np.ndarray] = None
        self._logits_np: Optional[np.ndarray] = None
        self._flags_np: Optional[np.ndarray] = None
        self._rnn_h_np: Optional[np.ndarray] = None
        self._rnn_c_np: Optional[np.ndarray] = None

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

        # Pre-cache numpy views (zero-copy) to avoid per-call allocation in _gather_obs
        self._obs_np = self.infer_obs.numpy()
        self._act_np = self.infer_act.numpy()
        self._logp_np = self.infer_logp.numpy()
        self._val_np = self.infer_val.numpy()
        self._flags_np = self.ready_flags.numpy()
        if self.use_lstm:
            self._mask_np = self.infer_mask.numpy()
            if self.rnn_state_live_h is not None:
                self._rnn_h_np = self.rnn_state_live_h.numpy()
                self._rnn_c_np = self.rnn_state_live_c.numpy()
        if self.infer_action_logits is not None:
            self._logits_np = self.infer_action_logits.numpy()

        # Initial weight load
        self.param_client.ensure_updated(self.policy)

        # Signal Manager that ZMQ sockets are bound and we're ready
        ev = self.ctx.ready_events.get(f"inference_server_{self.server_idx}")
        if ev is not None:
            ev.set()
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
        """Parse raw ZMQ messages into Nx2 int32 array: [flat_idx, op]."""
        buf = b"".join(raw_msgs)
        total_bytes = len(buf)
        if total_bytes == 0 or total_bytes % REQ_SIZE != 0:
            return np.empty((0, 2), dtype=np.int32)
        n = total_bytes // REQ_SIZE
        return np.frombuffer(buf, dtype=np.int32).reshape(n, 2)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _process_batch(self, requests: np.ndarray) -> None:
        ops = requests[:, 1]
        act_mask = ops == OP_ACT
        val_mask = ops == OP_VALUE

        if act_mask.any():
            self._run_act(requests[act_mask])
        if val_mask.any():
            self._run_value(requests[val_mask])

    def _gather_obs(self, flat_idxs: np.ndarray) -> Dict[str, Any]:
        """Gather obs from pre-cached numpy views using 1D int32 indexing.

        No astype conversion: flat_idxs is int32 from the ZMQ message, and
        numpy fancy indexing accepts int32 natively. torch.from_numpy wraps
        the result zero-copy (no extra allocation).
        """
        obs_np = self._obs_np[flat_idxs]          # int32 fancy index, returns copy
        obs_batch = torch.from_numpy(obs_np)       # zero-copy wrap
        if self.device.type != "cpu":
            obs_batch = obs_batch.to(self.device)

        inputs: Dict[str, Any] = {"obs": obs_batch}

        if self.use_lstm:
            if self._rnn_h_np is not None:
                h_np = self._rnn_h_np[flat_idxs]  # [N, hidden]
                c_np = self._rnn_c_np[flat_idxs]
                h = torch.from_numpy(h_np).unsqueeze(0)  # [1, N, hidden]
                c = torch.from_numpy(c_np).unsqueeze(0)
            else:
                h = torch.zeros(1, len(flat_idxs), self.policy.lstm_hidden_size)
                c = torch.zeros_like(h)
            if self.device.type != "cpu":
                h, c = h.to(self.device), c.to(self.device)
            inputs["rnn_state"] = (h, c)
            m_np = self._mask_np[flat_idxs]
            m = torch.from_numpy(m_np)
            if self.device.type != "cpu":
                m = m.to(self.device)
            inputs["mask"] = m

        return inputs

    def _run_act(self, reqs: np.ndarray) -> None:
        """Gather obs -> forward -> scatter to infer_* buffers -> set flags.

        reqs: Nx2 int32 array [flat_idx, op]
        """
        t_start = time.monotonic()

        flat_idxs = reqs[:, 0]  # int32, used directly for numpy indexing

        # Vectorized gather via pre-cached numpy views (no astype, no allocation)
        inputs = self._gather_obs(flat_idxs)
        t_gather = time.monotonic()
        self.prof.add("gather_obs", t_gather - t_start)

        # Forward pass
        out = self.policy.act(inputs, deterministic=False)
        t_fwd = time.monotonic()
        self.prof.add("forward", t_fwd - t_gather)

        # --- Scatter results via numpy (int32 native, no astype) ---
        actions = out["action"] if self.device.type == "cpu" else out["action"].cpu()
        self._act_np[flat_idxs] = actions.numpy()

        if "logp" in out:
            logps = out["logp"] if self.device.type == "cpu" else out["logp"].cpu()
            self._logp_np[flat_idxs] = logps.numpy()

        if "value" in out:
            values = out["value"] if self.device.type == "cpu" else out["value"].cpu()
            self._val_np[flat_idxs] = values.numpy()

        if "action_logits" in out and self._logits_np is not None:
            al = out["action_logits"] if self.device.type == "cpu" else out["action_logits"].cpu()
            self._logits_np[flat_idxs] = al.numpy()

        if self.use_lstm and "rnn_state" in out:
            h_out, c_out = out["rnn_state"]
            if self.device.type != "cpu":
                h_out, c_out = h_out.cpu(), c_out.cpu()
            h_out = h_out.squeeze(0)  # [N, hidden]
            c_out = c_out.squeeze(0)
            if self._rnn_h_np is not None:
                self._rnn_h_np[flat_idxs] = h_out.numpy()
                self._rnn_c_np[flat_idxs] = c_out.numpy()

        t_scatter = time.monotonic()
        self.prof.add("scatter", t_scatter - t_fwd)

        # Set ready flags (numpy int32 scatter)
        self._flags_np[flat_idxs] = 1

        t_signal = time.monotonic()
        self.prof.add("set_flags", t_signal - t_scatter)

    def _run_value(self, reqs: np.ndarray) -> None:
        """VALUE-only requests (bootstrap at trajectory boundary).

        reqs: Nx2 int32 array [flat_idx, op]
        """
        flat_idxs = reqs[:, 0]
        inputs = self._gather_obs(flat_idxs)

        v = self.policy.value(inputs)
        values = v if self.device.type == "cpu" else v.cpu()
        self._val_np[flat_idxs] = values.numpy()

        self._flags_np[flat_idxs] = 1


def main(ctx, logger_queue, server_idx: int = 0) -> None:
    child_sig_setup()
    child_logging_setup()
    child_attach_logger(logger_queue)
    logging.info("[inference:%d] starting", server_idx)
    InferenceServer(ctx, server_idx=server_idx).serve()
    logging.info("[inference:%d] stopped", server_idx)
