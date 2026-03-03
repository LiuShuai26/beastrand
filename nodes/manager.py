"""
Manager (v2): orchestrate all nodes.

Startup flow:
  1. Probe env → obs/act specs
  2. Create BufferMgr (shared tensors + queue)
  3. Create ParameterServer (shared weights + version + lock)
  4. Spawn DataServer, Learner, InferenceServer, Workers
  5. Monitor until target steps or crash
"""
from __future__ import annotations

import logging
import signal
import time
import multiprocessing as mp
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import gymnasium as gym

from modules.dataset.buffer_mgr import BufferMgr
from utils.import_utils import get_object_from_path
from utils.model_sharing import ParameterServer

from nodes.logger import start_logger, get_logger_queue
from nodes.data_server import main as data_server_main
from nodes.learner.learner import main as learner_main
from nodes.rollout_worker import main as worker_main
from nodes.inference_server import main as inference_server_main


def _now_s() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Env probing (unchanged)
# ---------------------------------------------------------------------------

_DTYPE2STR = {
    np.float32: "float32", np.float64: "float64", np.float16: "float16",
    np.int64: "int64", np.int32: "int32", np.uint8: "uint8", np.bool_: "bool",
}

def _dtype_str(dt: np.dtype) -> str:
    return _DTYPE2STR.get(np.dtype(dt).type, "float32")


def probe_env(env_id: str, seed: int = 0,
              make_env_path: Optional[str] = None,
              args=None) -> Dict[str, Any]:
    if make_env_path:
        _make_env = get_object_from_path(make_env_path)
        env = _make_env(env_id, seed=seed, args=args)
    else:
        env = gym.make(env_id)
    try:
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space

        if isinstance(obs_space, gym.spaces.Box):
            obs_shape = tuple(int(x) for x in obs_space.shape)
            obs_dtype = _dtype_str(obs_space.dtype)
        elif isinstance(obs_space, gym.spaces.Discrete):
            obs_shape = (1,)
            obs_dtype = "int64"
        else:
            raise NotImplementedError(f"Unsupported obs space: {type(obs_space)}")

        if isinstance(act_space, gym.spaces.Discrete):
            return {"obs": {"shape": list(obs_shape), "dtype": obs_dtype},
                    "act": {"kind": "discrete", "shape": [1], "dtype": "int64",
                            "n": int(act_space.n), "low": None, "high": None}}
        elif isinstance(act_space, gym.spaces.Box):
            result = {"obs": {"shape": list(obs_shape), "dtype": obs_dtype},
                      "act": {"kind": "box", "shape": list(int(x) for x in act_space.shape),
                              "dtype": "float32", "n": None,
                              "low": act_space.low, "high": act_space.high}}
            # Beast .so may expose AMP observation slices (single source of truth)
            if hasattr(env, "so_amp_obs_slices"):
                result["amp_obs_slices"] = env.so_amp_obs_slices
            return result
        else:
            raise NotImplementedError(f"Unsupported act space: {type(act_space)}")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

class Shared:
    """Lightweight pickleable context passed to all child processes."""

    def __init__(self, args):
        self.args = args
        self.run_name = getattr(args, "run_name", None) or f"{args.env_id}_{int(_now_s())}"
        self.start_time = _now_s()
        self.stop_event = mp.Event()
        self.global_step = mp.Value("i", 0)

        # Unique IPC directory for this run (avoids collisions between concurrent runs)
        self.ipc_dir = f"ipc:///tmp/beatstrand/{self.run_name}"

        # Filled by Manager.launch() before spawning children
        self.buffer_mgr: Optional[BufferMgr] = None
        self.param_server: Optional[ParameterServer] = None

        # Env specs (set by Manager)
        self.obs_shape = ()
        self.obs_dtype = "float32"
        self.act_kind = "box"
        self.act_shape = ()
        self.act_dtype = "float32"
        self.act_n = None
        self.act_low = None
        self.act_high = None

    def should_stop(self) -> bool:
        return self.stop_event.is_set()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class Manager:
    def __init__(self, args):
        self.args = args
        self.ctx = Shared(args)
        self.procs: Dict[str, mp.Process] = {}

    def _spawn(self, name: str, target: Callable, **kwargs) -> None:
        p = mp.Process(name=name, target=target, kwargs={"ctx": self.ctx, **kwargs})
        p.start()
        self.procs[name] = p
        logging.info("spawned %s [pid=%s]", name, p.pid)

    def _signal_all(self) -> None:
        self.ctx.stop_event.set()

    def _terminate_all(self) -> None:
        for name, p in self.procs.items():
            if p.is_alive():
                logging.info("terminating %s [pid=%s]", name, p.pid)
                p.terminate()

    def _join_all(self, timeout: Optional[float] = None) -> None:
        for name, p in self.procs.items():
            p.join(timeout=timeout)
            logging.info("joined %s [code=%s]", name, p.exitcode)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self) -> None:
        start_logger(
            logdir=getattr(self.args, "logdir", "train_logs"),
            experiment_name=self.ctx.run_name,
            queue_maxsize=30000,
            flush_every=2.0,
        )

        # 1. Probe env
        _mep = getattr(self.args, "make_env_path", None)
        specs = probe_env(self.args.env_id, seed=self.args.seed,
                          make_env_path=_mep, args=self.args)
        obs_spec, act_spec = specs["obs"], specs["act"]

        self.ctx.obs_shape = tuple(obs_spec["shape"])
        self.ctx.obs_dtype = obs_spec["dtype"]
        self.ctx.act_kind = act_spec["kind"]
        self.ctx.act_shape = tuple(act_spec["shape"])
        self.ctx.act_dtype = act_spec["dtype"]
        self.ctx.act_n = act_spec["n"]
        self.ctx.act_low = act_spec["low"]
        self.ctx.act_high = act_spec["high"]

        # Auto-detect AMP obs slices from Beast .so if available
        if "amp_obs_slices" in specs and hasattr(self.args, "amp_obs_slices"):
            so_slices = specs["amp_obs_slices"]
            cfg_slices = [tuple(s) for s in self.args.amp_obs_slices]
            if so_slices != cfg_slices:
                logging.warning(
                    "amp_obs_slices auto-updated from .so metadata: %s (was %s)",
                    so_slices, cfg_slices,
                )
                self.args.amp_obs_slices = so_slices
                self.args.amp_obs_dim = sum(e - s for s, e in so_slices)

        # 2. Compute topology
        num_workers = self.args.num_workers
        num_envs_per_worker = self.args.num_envs_per_worker
        T = self.args.rollout
        num_traj = num_workers * num_envs_per_worker

        # 3. Create BufferMgr (shared tensors + queue + async ready flags)
        buffer_mgr = BufferMgr(
            cfg=self.ctx,  # passes through to DataRecord.alloc_specs
            obs_shape=self.ctx.obs_shape,
            act_shape=self.ctx.act_shape,
            num_traj=num_traj,
            T=T,
            num_workers=num_workers,
            num_envs_per_worker=num_envs_per_worker,
        )
        self.ctx.buffer_mgr = buffer_mgr
        logging.info("BufferMgr: %d trajectories, T=%d", num_traj, T)

        # 4. Create ParameterServer (shared weights)
        device = torch.device(getattr(self.args, "device", "cpu"))
        policy_cls = get_object_from_path(self.args.policy_path)
        init_policy = policy_cls(self.ctx).to(device)
        param_server = ParameterServer(init_policy, device, buffer_mgr.policy_version)
        self.ctx.param_server = param_server
        del init_policy
        logging.info("ParameterServer: shared weights created")

        # 5. Spawn nodes
        log_q = get_logger_queue()

        self._spawn("data_server", data_server_main, logger_queue=log_q)
        time.sleep(0.5)  # let ZMQ sockets bind

        self._spawn("learner", learner_main, logger_queue=log_q)
        time.sleep(0.5)

        self._spawn("inference_server", inference_server_main, logger_queue=log_q)
        time.sleep(1.0)  # let inference server connect to req socket

        for i in range(num_workers):
            self._spawn(f"worker_{i}", worker_main, worker_idx=i, logger_queue=log_q)

        logging.info("All %d nodes spawned", len(self.procs))

    # ------------------------------------------------------------------
    # Run until target steps or crash
    # ------------------------------------------------------------------

    def run_until_complete(self) -> None:
        target_steps = self.args.total_env_steps
        logging.info("Target env steps: %d", target_steps)

        def handle_sig(sig, frame):
            logging.warning("Received signal %s — requesting shutdown...", sig)
            self._signal_all()

        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)

        try:
            last_log = _now_s()
            while not self.ctx.should_stop():
                if self.ctx.global_step.value >= target_steps:
                    logging.info("Reached target steps: %d", self.ctx.global_step.value)
                    break

                # Check for crashed processes
                for name, p in list(self.procs.items()):
                    if not p.is_alive() and p.exitcode not in (0, None):
                        logging.error("Process %s died with code %s — shutting down.", name, p.exitcode)
                        raise RuntimeError(f"process {name} died: {p.exitcode}")

                if _now_s() - last_log >= 5.0:
                    logging.info(
                        "heartbeat: steps=%d, version=%d, alive=%d/%d",
                        self.ctx.global_step.value,
                        int(self.ctx.buffer_mgr.policy_version.item()),
                        sum(int(p.is_alive()) for p in self.procs.values()),
                        len(self.procs),
                    )
                    last_log = _now_s()

                time.sleep(0.5)
        finally:
            self._signal_all()
            self._join_all(timeout=5.0)
            self._terminate_all()
            self._join_all(timeout=2.0)
            logging.info("Shutdown complete.")
