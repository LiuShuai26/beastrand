"""
Manager (v2): orchestrate all nodes.

Startup flow:
  1. Probe env → obs/act specs
  2. Validate module contracts (DataRecord ↔ Policy ↔ Algorithm)
  3. Create BufferMgr (shared tensors + split flags)
  4. Create ParameterServer (shared weights + version + lock)
  5. Spawn nodes with ready-event handshake (no sleep-based waits)
  6. Monitor until target steps or crash
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

from beastrand.core.buffer_mgr import BufferMgr
from beastrand.core.contract import validate_contracts
from beastrand.utils.import_utils import get_object_from_path
from beastrand.utils.model_sharing import ParameterServer

from beastrand.nodes.logger import start_logger, get_logger_queue
from beastrand.nodes.data_server import main as data_server_main
from beastrand.nodes.learner.learner import main as learner_main
from beastrand.nodes.rollout_worker import main as worker_main
from beastrand.nodes.inference_server import main as inference_server_main


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
            return {"obs": {"shape": list(obs_shape), "dtype": obs_dtype},
                    "act": {"kind": "box", "shape": list(int(x) for x in act_space.shape),
                            "dtype": "float32", "n": None,
                            "low": act_space.low, "high": act_space.high}}
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
        self.global_step = mp.Value("l", 0)  # int64: avoids overflow beyond ~2B steps

        # Unique IPC directory for this run (avoids collisions between concurrent runs)
        self.ipc_dir = f"ipc:///tmp/beatstrand/{self.run_name}"

        # Ready events: nodes set these after ZMQ bind completes
        self.ready_events: Dict[str, mp.Event] = {}

        # Filled by Manager.launch() before spawning children
        self.buffer_mgr: Optional[BufferMgr] = None
        self.param_server: Optional[ParameterServer] = None
        self.snapshot_dir: Optional[str] = None  # multi-agent: path to snapshot pool

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

    def _wait_ready(self, name: str, timeout: float = 10.0) -> None:
        ev = self.ctx.ready_events[name]
        if not ev.wait(timeout=timeout):
            raise RuntimeError(f"{name} did not become ready within {timeout}s")
        logging.info("%s ready", name)

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

    def launch(self, wandb_config: Optional[dict] = None) -> None:
        start_logger(
            logdir=getattr(self.args, "logdir", "train_logs"),
            experiment_name=self.ctx.run_name,
            queue_maxsize=30000,
            flush_every=2.0,
            wandb_project=getattr(self.args, "wandb_project", None),
            wandb_config=wandb_config,
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

        # 2. Validate module contracts (DataRecord ↔ Policy ↔ Algorithm)
        record_cls = get_object_from_path(self.args.data_record_path)
        policy_cls = get_object_from_path(self.args.policy_path)
        algorithm_cls = get_object_from_path(self.args.algorithm_path)
        validate_contracts(self.ctx, record_cls, policy_cls, algorithm_cls)

        # 3. Compute topology
        num_workers = self.args.num_workers
        num_envs_per_worker = self.args.num_envs_per_worker
        T = self.args.rollout

        # 4. Create BufferMgr (shared tensors + split flags)
        buffer_mgr = BufferMgr(
            cfg=self.ctx,  # passes through to DataRecord.alloc_specs
            obs_shape=self.ctx.obs_shape,
            act_shape=self.ctx.act_shape,
            T=T,
            num_workers=num_workers,
            num_envs_per_worker=num_envs_per_worker,
            split_depth=2,
        )
        self.ctx.buffer_mgr = buffer_mgr
        logging.info("BufferMgr: %d trajectories (split_depth=2), T=%d", buffer_mgr.num_traj, T)

        # 5. Create ParameterServer (shared weights, always CPU)
        init_policy = policy_cls(self.ctx)  # CPU — just need state_dict shape
        param_server = ParameterServer(init_policy, torch.device("cpu"), buffer_mgr.policy_version)
        self.ctx.param_server = param_server

        # 5b. Multi-agent: save initial snapshot so opponent IS can load immediately
        num_agents = getattr(self.args, "num_agents", 1)
        self_play_mode = getattr(self.args, "self_play_mode", "snapshot")
        if num_agents > 1 and self_play_mode in {"snapshot", "mixed"}:
            import os
            snap_dir = os.path.join(
                getattr(self.args, "logdir", "train_logs"),
                self.ctx.run_name, "snapshots",
            )
            os.makedirs(snap_dir, exist_ok=True)
            self.ctx.snapshot_dir = snap_dir
            from beastrand.utils.snapshot_pool import SnapshotPool
            SnapshotPool(snap_dir).save(init_policy, version=0)
            logging.info("Multi-agent: initial snapshot saved to %s", snap_dir)

        del init_policy
        logging.info("ParameterServer: shared weights created")

        # 6. Spawn nodes (with ready-event handshake)
        #
        # Nodes that bind ZMQ sockets signal readiness via mp.Event.
        # Manager waits for binders before spawning connectors.
        log_q = get_logger_queue()

        # Phase 1: DataServer (binds data.filled.in, data.filled.out)
        self.ctx.ready_events["data_server"] = mp.Event()
        self._spawn("data_server", data_server_main, logger_queue=log_q)
        self._wait_ready("data_server")

        # Phase 2: Learner (connects to data.filled.out — DataServer must be ready)
        self._spawn("learner", learner_main, logger_queue=log_q)

        # Phase 3: InferenceServers (bind infer_{i}.req)
        num_infer = getattr(self.args, "num_inference_servers", 1)
        for i in range(num_infer):
            name = f"inference_server_{i}"
            self.ctx.ready_events[name] = mp.Event()
            self._spawn(name, inference_server_main,
                        logger_queue=log_q, server_idx=i)
        for i in range(num_infer):
            self._wait_ready(f"inference_server_{i}")

        # Phase 3b: Per-agent InferenceServers for non-training agents (self-play)
        if num_agents > 1 and self_play_mode in {"snapshot", "mixed"}:
            for a in range(1, num_agents):
                name = f"inference_server_agent_{a}"
                self.ctx.ready_events[name] = mp.Event()
                self._spawn(name, inference_server_main,
                            logger_queue=log_q, server_idx=f"agent_{a}", agent_role=a)
            for a in range(1, num_agents):
                self._wait_ready(f"inference_server_agent_{a}")

        # Phase 4: Workers (connect to infer_{i}.req, data.filled.in — all binders ready)
        for i in range(num_workers):
            self._spawn(f"worker_{i}", worker_main, worker_idx=i, logger_queue=log_q)

        logging.info("All %d nodes spawned", len(self.procs))

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _run_eval(self) -> Optional[float]:
        """Run eval episodes with the latest policy weights. Returns mean episode reward."""
        from beastrand.utils.model_sharing import _load_state_into_model
        from beastrand.core.envs.make_env import make_env as _default_make_env

        n_episodes = getattr(self.args, "num_eval_episodes", 10)
        try:
            policy_cls = get_object_from_path(self.args.policy_path)
            device = torch.device("cpu")
            policy = policy_cls(self.ctx).to(device)
            _load_state_into_model(self.ctx.param_server.shared_state, policy)
            policy.eval()

            _mep = getattr(self.args, "make_env_path", None)
            _make_env = get_object_from_path(_mep) if _mep else _default_make_env

            rewards = []
            for ep in range(n_episodes):
                env = _make_env(self.args.env_id, seed=9999 + ep, args=self.args)
                obs, _ = env.reset(seed=9999 + ep)
                ep_reward = 0.0
                done = False
                steps = 0
                while not done and steps < 10_000:
                    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        out = policy.act({"obs": obs_t})
                    action = out["action"][0]
                    if self.ctx.act_kind == "discrete":
                        action = int(action.item())
                    else:
                        action = action.cpu().numpy()
                    obs, reward, term, trunc, _ = env.step(action)
                    ep_reward += float(reward)
                    done = bool(term or trunc)
                    steps += 1
                rewards.append(ep_reward)
                env.close()

            return float(np.mean(rewards))
        except Exception:
            logging.exception("[manager] eval failed")
            return None

    # ------------------------------------------------------------------
    # Run until target steps or crash
    # ------------------------------------------------------------------

    def run_until_complete(self) -> None:
        import threading
        from beastrand.nodes.logger import log_scalar

        target_steps = self.args.total_env_steps
        logging.info("Target env steps: %d", target_steps)

        eval_interval = getattr(self.args, "eval_interval", 0)
        _last_eval_step = 0
        _eval_thread: Optional[threading.Thread] = None
        _eval_results: list = []  # [(step, mean_reward)] written by eval thread

        def _launch_eval(step: int) -> None:
            def _worker():
                result = self._run_eval()
                if result is not None:
                    _eval_results.append((step, result))
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            return t

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

                # Periodic eval (background thread — does not block monitoring loop)
                if eval_interval > 0:
                    step = self.ctx.global_step.value
                    if step - _last_eval_step >= eval_interval:
                        if _eval_thread is None or not _eval_thread.is_alive():
                            _last_eval_step = step
                            _eval_thread = _launch_eval(step)

                # Log any completed eval results
                while _eval_results:
                    eval_step, mean_reward = _eval_results.pop(0)
                    n = getattr(self.args, "num_eval_episodes", 10)
                    log_scalar(run="eval", tag="episode_reward_mean", value=mean_reward, step=eval_step)
                    logging.info("[eval] step=%d mean_reward=%.2f (%d episodes)", eval_step, mean_reward, n)

                time.sleep(0.5)
        finally:
            self._signal_all()
            self._join_all(timeout=5.0)
            self._terminate_all()
            self._join_all(timeout=2.0)
            logging.info("Shutdown complete.")
