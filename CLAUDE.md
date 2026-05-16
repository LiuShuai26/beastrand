# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

beastrand is a high-throughput distributed PPO training framework — as simple as CleanRL, as fast as Sample Factory. ~3K lines of core code with a multi-process, node-based architecture using ZMQ IPC and PyTorch shared-memory tensors for zero-copy data sharing. Works with any Gymnasium env (MuJoCo, Atari, etc.) or custom C++ environments compiled as `.so` modules.

The core framework provides PPO; `projects/` contains example extensions (PPO-LSTM, PPO-AMP, MuJoCo, Atari) that demonstrate how to build on top of core without modifying it.

## Commands

**Run training:**
```bash
# PPO (default: Humanoid-v5)
python -m beastrand.ppo.train --env-id Humanoid-v5

# MuJoCo project (SF-matched hyperparams)
python -m projects.mujoco.train --env-id HalfCheetah-v4

# Atari project (SF-matched hyperparams, NatureCNN)
python -m projects.atari.train --env-id BreakoutNoFrameskip-v4

# PPO-LSTM
python -m projects.ppo_lstm.train --env-id Humanoid-v5

# PPO-AMP with Beast .so environment
python -m projects.ppo_amp.train --env-id HumanoidEnv --keyframe-file path/to/keyframes.json
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

The `Manager` (nodes/manager.py) is the orchestrator. It probes the env, validates module contracts (DataRecord ↔ Policy ↔ Algorithm), creates shared resources (BufferMgr, ParameterServer), spawns all nodes with ready-event handshake, and monitors until completion. Startup order: DataServer → Learner → InferenceServer → Workers. Binding nodes signal readiness via `mp.Event`; Manager waits on these before spawning connecting nodes.

**Nodes** (each runs in its own process):
- **Manager** — probes env, creates BufferMgr + ParameterServer, spawns and monitors all nodes
- **Logger** — daemon process, sole TensorBoard SummaryWriter
- **DataServer** — pure forwarder: routes filled trajectory IDs from workers to learner
- **InferenceServer** — zero-copy batched GPU inference via shared tensors + struct.pack ZMQ messages
- **Learner** — ingest thread (GAE computation) + main thread (PPO training), updates shared weights via ParameterServer
- **RolloutWorker** (×N) — each manages num_envs_per_worker environments with async per-env polling

### Shared memory: PyTorch share_memory_()

**BufferMgr** (beastrand/core/buffer_mgr.py) pre-allocates all trajectory tensors as `[num_traj, T+1, *shape]` PyTorch tensors with `share_memory_()`. Workers index into these via slot indices from a `mp.Queue`. No attach/detach, no extra file descriptors.

**ParameterServer** (beastrand/utils/model_sharing.py) holds a shared-memory copy of policy weights. Learner calls `param_server.update(policy)` after each training step. InferenceServer polls `policy_version` and loads when stale via `ParameterClient.ensure_updated()`.

### Communication patterns

**StrandBus** (beastrand/strandbus/strandbus.py) wraps ZMQ with named sockets. All IPC uses `ipc:///tmp/beatstrand/<run_name>/` endpoints (unique per run, enabling concurrent training):
- `infer.req` — PUSH/PULL for inference requests (struct.pack, 8 bytes: flat_idx, op)
- `data.filled.in` / `data.filled.out` — PUSH/PULL for filled trajectory IDs (struct.pack, 4 bytes)

Inference responses use shared memory flags (`ready_flags[worker_idx, env_idx]`) instead of ZMQ — zero-overhead async signaling. No pickle on the hot path.

### Data flow

1. BufferMgr creates `num_traj` trajectory slots as shared PyTorch tensors, queues all indices
2. Worker gets traj_idx from queue, writes obs to `traj_tensors["obs"][traj_idx, step]`
3. Worker sends struct request (8 bytes: flat_idx, op) to InferenceServer
4. InferenceServer gathers obs from shared tensors, runs forward pass, scatters action/logp/value back
5. InferenceServer sets `ready_flags[worker_idx, env_idx] = 1` (shared memory flag)
6. Worker polls ready_flags, reads action from shared tensor, does env.step (async — no blocking)
7. Worker writes reward/done, advances step. When step >= T: sends traj_idx to DataServer → Learner
8. Learner reads trajectory data (zero-copy numpy view), computes GAE, trains, puts traj_idx back in queue

### Projects (environment-specific configs)

Each project provides SF-matched hyperparameters, custom policy/env factories as needed:

- **`projects/mujoco/`** — MuJoCo continuous control (HalfCheetah, Humanoid, etc.). SF-matched hyperparams: lr=2.95e-3, batch=4096, epochs=2, mlp=[64,64].
- **`projects/atari/`** — Atari discrete control (Breakout, Pong, etc.). NatureCNN encoder, ClipRewardEnv, orthogonal init, SF-matched hyperparams.
- **`projects/ppo_lstm/`** — PPO with LSTM policy for partial observability.
- **`projects/ppo_amp/`** — PPO-AMP (Adversarial Motion Priors) for Beast .so environments. Discriminator + motion buffer for motion imitation.

### Environment backends

**Gymnasium (default)** — any env registered in the Gymnasium registry (e.g. `Humanoid-v5`, `CartPole-v1`), loaded via `gym.make()`.

The environment factory is pluggable via `make_env_path` in the config (dotted Python path). Projects can provide custom factories for non-Gymnasium environments. `probe_env()` in Manager also respects this path so that env specs are correctly probed for custom backends.

#### Multi-agent env contract

For multi-agent rollouts (`args.num_agents > 1`), the env passes non-training agents' observations and actions through attributes on the env object, not through the standard step/reset return values. The training agent (index 0) uses normal Gym semantics; other agents use side-channel attrs:

- `env.agent_obs: List[np.ndarray]` — populated by env on `reset()` / `step()`; index `a` is agent `a`'s observation.
- `env.agent_actions: List[np.ndarray]` — populated by the rollout worker before each `step()`; the env reads index `a` for agent `a`'s action.
- `env.num_agents: int` — total agent count, set in `__init__`.

`env.step(action)` takes only agent 0's action; the env reads `agent_actions[1:]` internally to drive the other agents and writes all agents' obs back into `agent_obs`. See `projects/tennis/pettingzoo_wrapper.py` for a reference implementation that wraps a PettingZoo parallel env into this contract.

### Module system (pluggable via dotted paths)

Config dataclasses (`Args`) specify `data_record_path`, `policy_path`, `algorithm_path`, and optionally `make_env_path` as dotted Python paths. These are resolved at runtime via `get_object_from_path()`.

Key module interfaces:
- **BasePolicy** (beastrand/core/base_policy.py) — `nn.Module`; must implement `act()`, optionally `value()`, `evaluate_actions()`
- **Algorithm** — protocol with `prepare_batch()` and `update()` (no base class; each algo is self-contained, CleanRL-style)
- **DataRecord** (beastrand/core/base_record.py) — defines `alloc_specs()` for tensor layout and `build_batch()` for training tensors

### Key conventions

- All child processes call `child_sig_setup()` (ignore SIGINT) and `child_logging_setup()` on start
- Shared entry-point boilerplate (`setup_logging`, `set_start_method`) lives in `beastrand/core/common.py`
- Shared node utilities (`ProfileAccum`, `child_sig_setup`, `child_logging_setup`) live in `beastrand/nodes/common.py`
- Logging uses a centralized logger process via `log_scalar()` → TensorBoard
- Process start method is `spawn` (required for CUDA safety)
- Torch sharing strategy is `file_system` (avoids fd limits)
- IPC endpoints are under `ipc:///tmp/beatstrand/<run_name>/` (unique per run, enabling concurrent training)
- StrandBus supports PUSH/PULL and PUB/SUB only (no ROUTER/DEALER)
- Minibatch iteration uses sequential slicing (no shuffle), matching SF's implementation
- Obs normalizer updates once per PPO iteration, then freezes for training epochs (SF-style)
