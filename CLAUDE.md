# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

beatstrand is a distributed reinforcement learning framework for high-throughput training. It uses a multi-process, node-based architecture with ZMQ IPC for communication and PyTorch shared-memory tensors for zero-copy data sharing. Implemented algorithms: PPO, PPO-LSTM, and PPO-AMP (Adversarial Motion Priors). PPO implementation referenced from CleanRL.

## Commands

**Run training:**
```bash
python -m run.run_ppo.train_ppo          # PPO (default: Humanoid-v5)
python -m run.run_ppo.train_ppo --env-id CartPole-v1 --num-workers 4

# Beast .so environment (e.g. HumanoidEnv)
python -m run.run_ppo_amp.train_ppo_amp --env-id HumanoidEnv --keyframe-file path/to/keyframes.json
python -m run.run_ppo.train_ppo --env-id HumanoidEnv --make-env-path modules.envs.make_env_amp.make_env_amp
```

CLI parsing uses `tyro` — all fields in the `Args` dataclass become CLI flags (use `--help` for full list).

**Run tests:**
```bash
python -m pytest tests/
python -m pytest tests/test_policy.py    # single test file
```

**View training logs:**
```bash
tensorboard --logdir train_logs/
```

## Architecture

### Node-based multi-process design

The `Manager` (nodes/manager.py) is the orchestrator. It creates shared resources (BufferMgr, ParameterServer), spawns all nodes, and monitors until completion. Startup order: DataServer → Learner → InferenceServer → Workers.

**Nodes** (each runs in its own process):
- **Manager** — probes env, creates BufferMgr + ParameterServer, spawns and monitors all nodes
- **DataServer** — pure forwarder: routes filled trajectory IDs from workers to learner
- **InferenceServer** — zero-copy batched GPU inference via shared tensors + struct.pack ZMQ messages
- **Learner** — ingest thread (GAE computation) + main thread (PPO training), updates shared weights via ParameterServer
- **RolloutWorker** (×N) — each manages num_envs_per_worker environments with async per-env polling (no splits)

### Shared memory: PyTorch share_memory_()

**BufferMgr** (modules/dataset/buffer_mgr.py) pre-allocates all trajectory tensors as `[num_traj, T+1, *shape]` PyTorch tensors with `share_memory_()`. Workers index into these via slot indices from a `mp.Queue`. No attach/detach, no extra file descriptors.

**ParameterServer** (utils/model_sharing.py) holds a shared-memory copy of policy weights. Learner calls `param_server.update(policy)` after each training step. InferenceServer polls `policy_version` and loads when stale via `ParameterClient.ensure_updated()`.

### Communication patterns

**StrandBus** (strandbus/strandbus.py) wraps ZMQ with named sockets. All IPC uses `ipc:///tmp/beatstrand/<run_name>/` endpoints (unique per run, enabling concurrent training):
- `infer.req` — PUSH/PULL for inference requests (struct.pack, 20 bytes)
- `data.filled.in` / `data.filled.out` — PUSH/PULL for filled trajectory IDs (struct.pack, 4 bytes)

Inference responses use shared memory flags (`ready_flags[worker_idx, env_idx]`) instead of ZMQ — zero-overhead async signaling. No pickle on the hot path.

### Data flow

1. BufferMgr creates `num_traj` trajectory slots as shared PyTorch tensors, queues all indices
2. Worker gets traj_idx from queue, writes obs to `traj_tensors["obs"][traj_idx, step]`
3. Worker sends struct request (20 bytes: traj_idx, step, worker_idx, env_idx, op) to InferenceServer
4. InferenceServer gathers obs from shared tensors, runs forward pass, scatters action/logp/value back
5. InferenceServer sets `ready_flags[worker_idx, env_idx] = 1` (shared memory flag)
6. Worker polls ready_flags, reads action from shared tensor, does env.step (async — no blocking)
7. Worker writes reward/done, advances step. When step >= T: sends traj_idx to DataServer → Learner
8. Learner reads trajectory data (zero-copy numpy view), computes GAE, trains, puts traj_idx back in queue

### Environment backends

Two environment backends are supported:

1. **Gymnasium (default)** — any env registered in the Gymnasium registry (e.g. `Humanoid-v5`, `CartPole-v1`), loaded via `gym.make()`.
2. **Beast .so** — custom compiled C++ environments (e.g. `HumanoidEnv.cpython-310-darwin.so`), loaded directly via `importlib.import_module` and wrapped with a built-in `BeastGymWrapper` (no external dependencies). The factory in `modules/envs/make_env_amp.py` tries Beast first; if the `.so` is not found, it falls back to `gym.make()`.

The environment factory is pluggable via `make_env_path` in the config (dotted Python path). `probe_env()` in Manager also respects this path so that env specs are correctly probed for custom backends.

### Module system (pluggable via dotted paths)

Config dataclasses (`Args`) specify `data_record_path`, `policy_path`, `algorithm_path`, and optionally `make_env_path` as dotted Python paths. These are resolved at runtime via `get_object_from_path()`.

Key module interfaces:
- **BasePolicy** (modules/policy/base_policy.py) — `nn.Module`; must implement `act()`, optionally `value()`, `evaluate_actions()`
- **Algorithm** — protocol with `prepare_batch()` and `update()` (no base class; each algo is self-contained, CleanRL-style)
- **DataRecord** (modules/dataset/data_record/) — defines `alloc_specs()` for tensor layout and `build_batch()` for training tensors

### Key conventions

- All child processes call `child_sig_setup()` (ignore SIGINT) and `child_logging_setup()` on start
- Shared entry-point boilerplate (`setup_logging`, `set_start_method`) lives in `run/common.py`
- Shared node utilities (`ProfileAccum`, `child_sig_setup`, `child_logging_setup`) live in `nodes/common.py`
- Logging uses a centralized logger process via `log_scalar()` → TensorBoard
- Process start method is `spawn` (required for CUDA safety)
- Torch sharing strategy is `file_system` (avoids fd limits)
- IPC endpoints are under `ipc:///tmp/beatstrand/<run_name>/` (unique per run, enabling concurrent training)
- StrandBus supports PUSH/PULL and PUB/SUB only (no ROUTER/DEALER)
