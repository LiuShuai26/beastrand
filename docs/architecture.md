# beastrand Architecture

This document is a deep dive into beastrand's internals: data flow, shared resources, design decisions, and trade-offs. Read the [README](../README.md) first for the high-level overview.

## Process Model

beastrand uses a multi-process architecture with 5 node types, all spawned by the Manager:

```
Manager (main process)
├── Logger              daemon, TensorBoard writer
├── DataServer          trajectory ID forwarder
├── Learner             GAE + PPO training (2 threads)
├── InferenceServer(s)  batched GPU inference
└── Worker × N          env stepping + inference requests
```

**Startup order matters**: downstream nodes (that bind ZMQ sockets) start before upstream nodes (that connect). Sleeps between spawns give ZMQ time to bind.

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

`num_traj = num_workers × num_envs_per_worker`. Each env occupies one slot at a time.

**Slot allocation** uses a token pool (`mp.Queue`):

```
Worker gets slot_idx from queue      → "I'll write to slot 3"
Worker fills T steps of data
Worker sends slot_idx to Learner     → "Slot 3 is ready for training"
Learner trains on slot 3, returns it → queue.put(3)
```

No locks needed — at any moment, only one process holds a given slot_idx.

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

`torch.zeros(num_workers, num_envs_per_worker, dtype=int32).share_memory_()`

InferenceServer writes `1` after scattering action into shared tensors. Workers poll their flag and reset to `0` after reading. Faster than ZMQ reply — action data is already in shared memory, only a 1-bit signal is needed.

## Data Flow

### Hot Path (Per Step)

```
Worker                    InferenceServer              Shared Memory
  │                            │                           │
  ├─ write obs[ti,s] ──────────────────────────────────► obs tensor
  ├─ send 20B struct ────────►│                           │
  │  (ti, step, wi, ei, op)   │                           │
  │                            ├─ gather obs ◄──────── obs tensor
  │                            ├─ GPU forward (batched)    │
  │                            ├─ scatter action ────────► action tensor
  │                            ├─ scatter logp ──────────► logp tensor
  │                            ├─ scatter value ─────────► value tensor
  │                            ├─ set flag ──────────────► ready_flags
  ├─ poll flag ◄───────────────────────────────────────── ready_flags
  ├─ read action ◄─────────────────────────────────────── action tensor
  ├─ env.step(action)          │                           │
  └─ write reward, done ───────────────────────────────► reward, done tensors
```

**Zero pickle, zero serialization.** ZMQ only carries 20-byte index messages.

### Trajectory Completion Path

```
Worker ──4B traj_idx──► DataServer ──forward──► Learner (ingest thread)
                                                  │
                                                  ├─ slot_as_numpy(ti)    zero-copy view
                                                  ├─ prepare_batch()      GAE computation
                                                  ├─ append to BatchBuffer (copy)
                                                  ├─ recycle: queue.put(ti)
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

**Why not double buffering?** With async per-env polling, CPU/GPU pipelining happens naturally at the finest granularity (each env independently). Double buffering (grouping envs into splits) would introduce unnecessary intra-group synchronization. Benchmarks confirmed no throughput gain (see [docs/design/remove-worker-splits.md](design/remove-worker-splits.md)).

**Note**: Double buffering IS useful when `env.step()` runs on GPU (e.g., Isaac Gym), where step and inference compete for the same hardware and need CUDA stream-level overlap. beastrand's CPU envs don't have this problem.

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
│  recycle slot_idx           checkpoint       │
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

**Request parsing**: All messages are concatenated into one `bytes` buffer, then `np.frombuffer(...).reshape(N, 5)` parses them in one operation. No per-message `struct.unpack`.

## Module System

Four pluggable extension points, specified as dotted Python paths in the config:

| Module | Interface | Purpose |
|--------|-----------|---------|
| `policy_path` | `BasePolicy` (act, value, evaluate_actions) | Neural network architecture |
| `algorithm_path` | `prepare_batch()`, `update()` | Training algorithm |
| `data_record_path` | `alloc_specs()`, `build_batch()` | Shared tensor layout |
| `make_env_path` | `make_env(env_id, seed, args)` | Environment factory |

Resolved at runtime via `get_object_from_path()` (6-line importlib wrapper). To add a new variant: implement the interface, set the path in config. No framework code changes.

**Trade-off**: No IDE jump-to-definition for dynamic paths, and typos only surface at runtime. But adding PPO-LSTM or PPO-AMP required zero changes to the core infrastructure.

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
| Core code | ~300 LOC/algo | ~50K LOC | ~30K LOC | ~200K+ LOC | ~2K LOC |
| Process model | Single | SubprocVecEnv | Multi-process async | Ray actors | Multi-process async |
| Data transfer | In-memory | Pickle over pipe | Shared memory | Ray object store | Shared memory + ZMQ |
| GPU utilization | ~10% | ~20-30% | ~60-80% | ~40-60% | ~60-70% |
| Multi-machine | No | No | No | Yes (Ray) | No (tcp:// ready) |
| Algorithms | Many | 6+ | PPO/APPO | 30+ | PPO family |
| AMP support | No | No | No | No | Native |
| Read full codebase | 1 hour | Impractical | Days | Impractical | Half a day |

beastrand and Sample Factory (default config, without V-trace) implement the same algorithm: async-collected standard PPO with clipped ratio. The difference is engineering: 2K lines vs 30K lines for comparable throughput.

## Code Organisation: Framework vs Projects

beastrand has three layers of code with a strict dependency direction: `projects → ppo → core/nodes`.

```
beastrand/
├── nodes/                     Infrastructure: process topology, communication
├── strandbus/                 Infrastructure: ZMQ IPC wrapper
├── utils/                     Infrastructure: param server, import, tensor utils
├── core/                      Interfaces + reusable building blocks
│   ├── base_policy.py           Policy interface (act, value, evaluate_actions)
│   ├── base_record.py           DataRecord interface (alloc_specs, build_batch)
│   ├── model/                   MLP, distributions
│   └── envs/make_env.py         Standard Gymnasium factory
│
├── ppo/                       Standard equipment — the default algorithm
│   ├── policy.py                Actor-critic with shared MLP body
│   ├── algorithm.py             GAE + PPO clipped update
│   ├── data_record.py           Trajectory tensor layout
│   ├── config.py                Hyperparameters (tyro CLI)
│   └── train.py                 Entry point
│
├── projects/                  Extensions built on top of PPO
│   ├── ppo_lstm/                PPO + recurrent policy (truncated BPTT)
│   └── ppo_amp/                 PPO + adversarial motion priors
```

**Infrastructure** (nodes, strandbus, utils) is algorithm-agnostic — it handles process spawning, ZMQ messaging, shared-memory tensors, weight sync, and batched GPU inference. None of this code knows about PPO.

**PPO** is the framework's default algorithm. It lives in its own directory (not in `projects/`) because all current extensions build on it — PPO-LSTM and PPO-AMP both import `compute_gae` and inherit `PPODataRecord`.

**Projects** are algorithm extensions or application-specific code. They depend on `ppo/` and `core/`, never the other way around.

## Adding New Algorithms

The infrastructure (Manager, InferenceServer, BufferMgr, ParameterServer, StrandBus) is algorithm-agnostic. Adding a new on-policy algorithm (e.g., A2C) means creating a new top-level directory with `policy.py`, `algorithm.py`, `data_record.py`, `config.py`, and `train.py` — no changes to `nodes/` or `core/`. For off-policy algorithms (SAC, DQN), the main adaptation is in the Learner: instead of consuming each trajectory once, transitions are copied into a replay buffer and slots are recycled immediately. BufferMgr and InferenceServer require no changes.
