# beastrand Architecture

This document is a deep dive into beastrand's internals: data flow, shared resources, design decisions, and trade-offs. Read the [README](../README.md) first for the high-level overview.

## Process Model

beastrand uses a multi-process architecture with 6 node types. Manager is the main process that spawns the rest:
```
Manager (main process)
├── Logger              daemon, TensorBoard writer
├── DataServer          trajectory ID forwarder
├── Learner             GAE + PPO training (2 threads)
├── InferenceServer(s)  batched GPU inference
└── Worker × N          env stepping + inference requests
```

**Startup order matters**: downstream nodes (that bind ZMQ sockets) start before upstream nodes (that connect). Each binding node signals readiness via `mp.Event`; Manager waits on these events before spawning connecting nodes (no sleep-based waits).

### Why Multi-Process (not Multi-Thread)?

- **GIL bypass**: Each Worker truly parallelizes `env.step()` on different CPU cores
- **CUDA safety**: `spawn` start method gives each process a clean GPU context
- **Fault isolation**: A segfault in one Worker doesn't crash the Learner

**Cost**: Inter-process communication requires ZMQ + shared memory, more complex than thread-local variables.

## Shared Resources

All shared resources are created by Manager before spawning any child process.

### BufferMgr

Pre-allocates all trajectory tensors in shared PyTorch memory (`Tensor.share_memory_()`). Child processes inherit references — no attach/detach, no extra file descriptors.

```
traj_tensors["obs"]     →  [num_traj, T+1, obs_dim]    float32
traj_tensors["action"]  →  [num_traj, T,   act_dim]    float32
traj_tensors["reward"]  →  [num_traj, T]                float32
traj_tensors["done"]    →  [num_traj, T]                uint8
traj_tensors["value"]   →  [num_traj, T+1]              float32
...
```

`num_traj = num_workers × num_envs_per_worker × split_depth` (split_depth=2).

**Slot assignment** is deterministic:

```
flat_idx = worker_idx * num_envs_per_worker + env_idx
traj_idx = flat_idx * split_depth + split_idx   (split_idx cycles 0→1→0→...)
```

Workers cycle `split_idx` each trajectory. The Learner releases the slot via `split_flags[flat_idx, split_idx] = 0`; the worker spin-waits on this flag before reusing the slot (backpressure, replaces `mp.Queue`). No IPC on the hot path.

**Inference I/O buffers** are separate flat `[W*E, *shape]` shared tensors (`infer_obs`, `infer_act`, `infer_logp`, `infer_val`). Workers write current obs here before the ZMQ request; IS reads/writes these compact buffers instead of the large trajectory tensors. See [inference-hotpath-optimization.md](design/inference-hotpath-optimization.md) for the full optimization history.

**Self-play buffers**: when `num_agents > 1`, BufferMgr allocates two extra per-opponent-agent buffer families:

- `latest_policy_infer`: obs/act/ready buffers served by the training-policy InferenceServer. Used for `latest` self-play, where agent 0 and agent 1+ are evaluated by the same current policy in one batched forward pass.
- `snapshot_infer`: obs/act/ready buffers served by dedicated frozen-opponent InferenceServers. Used for `snapshot` self-play, where the opponent is sampled from the snapshot pool.

When `self_play_mode="mixed"`, Workers route each episode to one of these two paths at reset time and keep that choice fixed for the full episode.

**Tensor layout** is defined by DataRecord (pluggable per algorithm). BufferMgr calls `DataRecord.alloc_specs()` to learn what fields to allocate. This way BufferMgr doesn't know about algorithm-specific fields (e.g., PPO-AMP's `amp_transition`).

### ParameterServer

Holds a shared-memory copy of policy weights (always CPU) for Learner → InferenceServer sync:

```
Learner:           param_server.update(policy)          # GPU → CPU shared mem, version++
InferenceServer:   param_client.ensure_updated(policy)  # check version, CPU shared mem → GPU
```

Uses a single `mp.Lock` to prevent weight tearing. Lock contention is minimal because updates are infrequent (once per training step, ~tens of ms).

**Optimistic version check**: InferenceServer reads `policy_version` (atomic int64) without locking. Only if version changed does it acquire the lock and copy weights.

### ready_flags

`torch.zeros(num_workers * num_envs_per_worker, dtype=int32).share_memory_()`

Flat `[W*E]` layout indexed by `flat_idx = worker_idx * E + env_idx`. InferenceServer writes `1` after scattering results into infer_* buffers. Workers poll and reset to `0` after reading. Faster than ZMQ reply — result data is already in shared memory, only a 1-bit signal is needed.

## Data Flow

### Hot Path (Per Step)

```
Worker                    InferenceServer              Shared Memory
  │                            │                           │
  ├─ write obs[ti,s] ──────────────────────────────────► traj_tensors["obs"]
  ├─ write infer_obs[flat_idx] ────────────────────────► infer_obs [W*E, obs_dim]
  ├─ send 12B struct ────────►│                           │
  │  (flat_idx, op, agent)     │                           │
  │                            ├─ gather obs ◄──────── infer_obs (numpy 1D)
  │                            ├─ GPU forward (batched)    │
  │                            ├─ scatter action ────────► infer_act [W*E, act_dim]
  │                            ├─ scatter logp ──────────► infer_logp [W*E]
  │                            ├─ scatter value ─────────► infer_val [W*E]
  │                            ├─ set flag ──────────────► ready_flags [W*E]
  ├─ poll flag ◄───────────────────────────────────────── ready_flags[flat_idx]
  ├─ read action ◄─────────────────────────────────────── infer_act[flat_idx]
  ├─ write action/logp/val to traj_tensors ────────────► traj_tensors["action"]...
  ├─ env.step(action)          │                           │
  └─ write reward, done ───────────────────────────────► traj_tensors["reward"]...
```

**Zero pickle, zero serialization.** ZMQ only carries a 12-byte `(flat_idx, op, agent_slot)` header:

- `flat_idx`: worker/env slot
- `op`: ACT or VALUE request
- `agent_slot`: `0` for the training agent, `1..N-1` for `latest` self-play opponents served by the same InferenceServer

### Trajectory Completion Path

```
Worker ──4B traj_idx──► DataServer ──forward──► Learner (ingest thread)
                                                  │
                                                  ├─ slot_as_numpy(ti)    zero-copy view
                                                  ├─ prepare_batch()      GAE computation
                                                  ├─ append to BatchBuffer (copy)
                                                  ├─ release: split_flags[flat,split]=0
                                                  │
                                                  ▼
                                                Learner (main thread)
                                                  ├─ build_batch()        numpy → GPU tensor
                                                  ├─ algorithm.update()   PPO training
                                                  └─ param_server.update()  sync weights
```

### Weight Sync Path

```
Learner                  ParameterServer              InferenceServer
  │                         (CPU shared mem)                │
  ├─ optimizer.step()        │                              │
  ├─ update() ──────────► copy weights, version++           │
  │                         │                              │
  │                         │ ◄─── ensure_updated() ───────┤
  │                         │      (polls version,          │
  │                         │       loads if changed)       │
```

## Per-Env Async Polling

Each Worker manages `num_envs_per_worker` environments. The main loop polls all envs every iteration:

```python
for es in self.envs:
    if not es.pending:     continue  # not waiting for inference
    if not ready_flag:     continue  # result not back yet
    advance(es)                      # step env, write data
    send_request(es)                 # request next inference
```

This is event-driven, not round-robin. Whichever env's inference result arrives first gets stepped first. CPU is never idle waiting for a single env.

**Trajectory-level double buffering (`split_depth=2`)**: Each (worker, env) pair owns 2 trajectory slots, cycling between them. While the Learner processes slot 0, the Worker fills slot 1 — pure producer-consumer overlap with no synchronization except the spin-wait at trajectory boundaries. This is not step-level double buffering.

**Why not step-level double buffering?** Step-level double buffering groups envs into two alternating batches to pipeline inference and env.step. With async per-env polling, CPU/GPU pipelining happens naturally at the finest granularity (each env independently), so adding batch-level grouping would introduce unnecessary intra-group synchronization with no throughput gain.

**Note**: Step-level double buffering IS useful when `env.step()` runs on GPU (e.g., Isaac Gym), where step and inference compete for the same hardware and need CUDA stream-level overlap. beastrand's CPU envs don't have this problem.

## Learner: Dual-Thread Design

```
┌──────────────────────────────────────────────┐
│                 Learner Process               │
│                                              │
│  Ingest Thread              Main Thread      │
│  ─────────────              ───────────      │
│  recv traj_idx              wait(Condition)  │
│  zero-copy numpy view       build_batch()    │
│  prepare_batch (GAE)  ───►  PPO update (GPU) │
│  append BatchBuffer  notify param_server     │
│  release split_flags        checkpoint       │
└──────────────────────────────────────────────┘
```

**Why two threads?** Ingest (IO + CPU numpy) and training (GPU) can overlap. While the main thread runs PPO on the current batch, the ingest thread is already receiving and processing the next batch.

Synchronization: one `threading.Lock` protects BatchBuffer, one `threading.Condition` notifies the main thread when enough data is ready.

## InferenceServer: Batch Formation

`recv_many()` blocks until the first message arrives, then non-blocking drains everything in the ZMQ queue. This naturally forms large batches:

```
During GPU forward (e.g., 2ms):
  Worker requests accumulate in ZMQ queue
After forward returns:
  recv_many() collects all queued requests → large batch
```

Higher load → more requests queue up → bigger batches → better GPU utilization. Positive feedback loop.

**Request parsing**: All messages are concatenated into one `bytes` buffer, then `np.frombuffer(...).reshape(N, 3)` parses them in one operation (Nx3 int32: `[flat_idx, op, agent_slot]`). No per-message `struct.unpack`. Numpy pre-cached views eliminate `astype(int64)` and `torch.from_numpy` overhead on the gather path.

## Self-Play Data Flow

Self-play is project-level behavior layered on top of the same worker / learner / inference topology. The framework supports three opponent modes:

- `latest`: both sides use the current training policy
- `snapshot`: the learning agent uses the current policy; the opponent uses a frozen historical snapshot
- `mixed`: per episode, sample `latest` with probability `latest_self_play_ratio` and `snapshot` otherwise

### Episode-Level Opponent Sampling

Opponent selection happens in the Worker, not inside the InferenceServer. Each env chooses its opponent mode at reset time and keeps that mode fixed until the episode ends.

This is important for PPO correctness: a trajectory should correspond to interaction against one stationary opponent policy, not an opponent that changes midway through the rollout.

### `latest` Self-Play

In `latest` mode, both sides' observations are sent to the training-policy InferenceServer:

```
Worker
  ├─ agent0 obs ─► infer_obs
  ├─ agent1 obs ─► latest_policy_infer[0].obs
  ├─ send (flat, ACT, 0) ─────► training IS
  └─ send (flat, ACT, 1) ─────► training IS

InferenceServer
  ├─ gather mixed batch: [agent0 rows, agent1 rows, ...]
  ├─ single forward pass on current policy
  ├─ scatter agent0 action/logp/value → infer_*
  └─ scatter agent1 action           → latest_policy_infer[*]
```

The key property is that both sides are evaluated by the **same policy version** in one batched forward pass. This avoids policy skew between the two sides and gives the expected "latest-vs-latest" self-play semantics.

### `snapshot` Self-Play

In `snapshot` mode, agent 0 still uses the training-policy InferenceServer, but opponent agents use dedicated snapshot InferenceServers that periodically load frozen checkpoints from the snapshot pool.

```
agent0 obs ─► training IS
agent1 obs ─► snapshot IS
```

Only agent 0's trajectory is recorded for learning. Opponent actions affect the environment but do not contribute log-probs, values, or learner batches.

### `mixed` Self-Play

`mixed` mode combines the two paths above:

- sample `latest` with probability `latest_self_play_ratio`
- sample `snapshot` with probability `1 - latest_self_play_ratio`
- hold that choice fixed for the full episode

Example: with `latest_self_play_ratio=0.7`, approximately 70% of episodes are `latest-vs-latest` and 30% are played against frozen historical opponents.

## Module System

Four pluggable extension points, specified as dotted Python paths in the config:

| Module | Interface | Purpose |
|--------|-----------|---------|
| `policy_path` | `BasePolicy` (act, value, evaluate_actions) | Neural network architecture |
| `algorithm_path` | `prepare_batch()`, `update()` | Training algorithm |
| `data_record_path` | `alloc_specs()`, `build_batch()` | Shared tensor layout |
| `make_env_path` | `make_env(env_id, seed, args)` | Environment factory |

Resolved at runtime via `get_object_from_path()` (6-line importlib wrapper). To add a new variant: implement the interface, set the path in config. No framework code changes.

**Contract validation**: At startup, Manager calls `validate_contracts()` (core/contract.py) which verifies the three-way contract between DataRecord, Policy, and Algorithm: `alloc_specs()` → fake view → `build_batch()` must produce valid arrays, and `policy.act()` must return `"action"`. Mismatches (e.g., typo'd key in a new project's DataRecord) fail at startup with a clear error, not minutes into training.

**Trade-off**: No IDE jump-to-definition for dynamic paths. But contract validation catches typos at startup, and adding PPO-LSTM or PPO-AMP required zero changes to the core infrastructure.

## PPO Algorithm

**On-policy with async collection**: Uses standard PPO clipped ratio, not V-trace. This works well when policy lag is small. `max_policy_lag` discards excessively stale trajectories as a safety net. If scaling to hundreds of workers where lag grows large, consider adding V-trace (see Sample Factory's `--with_vtrace` for reference).

## Communication: StrandBus

Thin ZMQ wrapper providing named sockets:

```python
bus.open("infer_req", mode="push", endpoint="ipc:///tmp/beatstrand/run1/infer_0.req", bind=False)
bus.send("infer_req", data)
msgs = bus.recv_many("filled_in")
```

- **Only IPC** (`ipc://`) currently. Switching to `tcp://` for multi-machine is a localized change.
- **PUSH/PULL** for point-to-point (inference requests, filled trajectories)
- **PUB/SUB** available but unused currently
- **Unique IPC paths** per run (`/tmp/beatstrand/{run_name}/`), enabling concurrent training runs

## Logger

Separate daemon process holding the sole `SummaryWriter`. All other processes send `_MsgScalar` via `mp.Queue`:

```python
log_scalar(run="learner", tag="pi_loss", value=0.05, step=100)
# → TensorBoard: learner/pi_loss = 0.05 @ step 100
```

**Drop-on-full** (`put_timeout=2ms`): If the queue is full, the message is silently dropped. Training throughput is never blocked by logging.

## Comparison with Other Frameworks

|  | CleanRL | SB3 | Sample Factory | RLlib | beastrand |
|--|---------|-----|----------------|-------|-----------|
| Core code | ~300 LOC/algo | ~50K LOC | ~30K LOC | ~200K+ LOC | ~3K LOC |
| Process model | Single | SubprocVecEnv | Multi-process async | Ray actors | Multi-process async |
| Data transfer | In-memory | Pickle over pipe | Shared memory | Ray object store | Shared memory + ZMQ |
| GPU utilization | ~10% | ~20-30% | ~60-80% | ~40-60% | ~65-75% |
| Multi-machine | No | No | No | Yes (Ray) | No (tcp:// ready) |
| Algorithms | Many | 6+ | PPO/APPO | 30+ | PPO family |
| AMP support | No | No | No | No | Example project |
| Read full codebase | 1 hour | Impractical | Days | Impractical | Half a day |

beastrand and Sample Factory (default config, without V-trace) implement the same algorithm: async-collected standard PPO with clipped ratio. The difference is engineering: 3K lines vs 30K lines for comparable throughput.

## Code Organisation: Framework vs Projects

beastrand has three layers of code with a strict dependency direction: `projects → ppo → core/nodes`.

```
beastrand-repo/                    ← git repo root
├── beastrand/                     ← pip package (pip install -e .)
│   ├── nodes/                       Infrastructure: process topology, communication
│   ├── strandbus/                   Infrastructure: ZMQ IPC wrapper
│   ├── utils/                       Infrastructure: param server, import, tensor utils
│   ├── core/                        Interfaces + reusable building blocks
│   │   ├── base_policy.py             Policy interface (act, value, evaluate_actions)
│   │   ├── base_record.py             DataRecord interface (alloc_specs, build_batch)
│   │   ├── model/                     MLP, distributions
│   │   └── envs/make_env.py           Standard Gymnasium factory
│   │
│   └── ppo/                         Standard equipment — the default algorithm
│       ├── policy.py                  Actor-critic with shared MLP body
│       ├── algorithm.py               GAE + PPO clipped update
│       ├── data_record.py             Trajectory tensor layout
│       ├── config.py                  Hyperparameters (tyro CLI)
│       └── train.py                   Entry point
│
└── projects/                      ← NOT part of the package (like sf_examples/)
    ├── mujoco/                      MuJoCo presets (SF-matched hyperparams)
    ├── atari/                       Atari presets (NatureCNN, ClipRewardEnv)
    ├── ppo_lstm/                    PPO + recurrent policy (truncated BPTT)
    └── ppo_amp/                     PPO + adversarial motion priors
```

**Infrastructure** (nodes, strandbus, utils) is algorithm-agnostic — it handles process spawning, ZMQ messaging, shared-memory tensors, weight sync, and batched GPU inference. None of this code knows about PPO.

**PPO** is the framework's default algorithm. It lives inside the package because all current extensions build on it — PPO-LSTM and PPO-AMP both import `compute_gae` and inherit `PPODataRecord`.

**Projects** live at the repo root (not inside the pip package), mirroring Sample Factory's `sf_examples/` layout. They are runnable from the repo root via `python -m projects.mujoco.train`. They depend on `beastrand.ppo` and `beastrand.core`, never the other way around.

## Adding New Algorithms

The infrastructure (Manager, InferenceServer, BufferMgr, ParameterServer, StrandBus) is algorithm-agnostic. Adding a new on-policy algorithm (e.g., A2C) means creating a new top-level directory with `policy.py`, `algorithm.py`, `data_record.py`, `config.py`, and `train.py` — no changes to `nodes/` or `core/`. For off-policy algorithms (SAC, DQN), the main adaptation is in the Learner: instead of consuming each trajectory once, transitions are copied into a replay buffer and slots are recycled immediately. BufferMgr and InferenceServer require no changes.
