# Inference Hot Path Optimization

Two rounds of optimization to the IS (InferenceServer) hot path, driven by profiling.

## Baseline (v3)

Original data flow per inference request:

```
Worker → ZMQ 20B (traj_idx, step, worker_idx, env_idx, op)
IS: Python for-loop over N requests
    for ti, s in zip(traj_idxs, steps):
        obs_list.append(traj_tensors["obs"][ti, s])  # random scatter into [num_traj, T+1, obs_dim]
    obs_batch = torch.stack(obs_list)                # N separate tensor reads
```

**Profiling** (8 workers × 2 envs, Humanoid-v4):

| Phase | Time |
|-------|------|
| gather_obs | 0.54ms |
| scatter results | 0.43ms |
| forward | ~0.9ms |
| **total/call** | **1.87ms** |
| SPS | ~7,400 |

Bottleneck: Python `for`-loop over scattered `traj_tensors[num_traj, T+1, *shape]` — O(N) PyTorch index dispatches, cold cache lines from large tensor.

---

## Plan B: Compact Inference I/O Buffers (v4)

**Key idea**: add a compact `[W, E, *shape]` shared buffer sitting between Worker and IS. Workers write current obs here before sending the ZMQ request. IS reads from this small buffer (fits in L2 cache) instead of gathering from the large trajectory tensor.

**Changes**:
- `BufferMgr`: add `infer_obs/act/logp/val/mask/action_logits [W, E, *shape]` shared tensors
- Workers: write obs to `infer_obs[wi, ei]` (in addition to `traj_tensors["obs"][ti, s]`)
- IS: replace Python for-loop with single vectorized index `infer_obs[wi, ei]` (torch fancy indexing)
- ZMQ message: drop `traj_idx` and `step` → `(worker_idx, env_idx, op)` = **12 bytes**
- IS scatter: similarly vectorized to `infer_act/logp/val[wi, ei]`

**Profiling**:

| Phase | Baseline | Plan B | Change |
|-------|----------|--------|--------|
| gather_obs | 0.54ms | 0.42ms | −22% |
| scatter | 0.43ms | 0.20ms | −53% |
| forward | ~0.9ms | ~0.9ms | — |
| **total/call** | **1.87ms** | **1.59ms** | **−15%** |
| SPS | 7,400 | 8,600 | +16% |

Scatter improved more than gather because the old scatter wrote to `traj_tensors[num_traj, T+1, *shape]` (exclusive cache line ownership for many scattered addresses). Gather only improved 22% because `astype(int64)` + `from_numpy` Python dispatch still dominated.

---

## Plan C: Fixed Traj Allocation + Flat Numpy Gather (v5)

**Motivation from Sample Factory analysis**: SF achieves fast gather by organizing trajectory slots contiguously per worker and using slice indexing. The deeper insight: SF's approach works because slots are assigned deterministically, not pulled from a random pool.

**Key idea**: replace `mp.Queue` slot pool with fixed deterministic assignment:

```
flat_idx = worker_idx * num_envs_per_worker + env_idx
traj_idx = flat_idx * split_depth + split_idx
```

Workers cycle `split_idx` (0, 1, 0, 1, ...) each trajectory. The Learner releases a slot by setting `split_flags[flat_idx, split_idx] = 0`.

**Changes**:

### BufferMgr
- Remove `traj_buffer_queue` (mp.Queue)
- Add `split_flags [W*E, split_depth]` int8 shared tensor (0=worker-owned, 1=learner-in-use)
- Flatten all inference buffers: `[W, E, *shape]` → `[W*E, *shape]`
- Flatten `ready_flags`: `[W, E]` → `[W*E]`
- Flatten `rnn_state_live_h/c`: `[W, E, hidden]` → `[W*E, hidden]`

### RolloutWorker
- Compute `traj_idx` locally (no queue.get())
- Track `_env_split[ei]` per env (local numpy int32 array)
- Spin-wait on `split_flags[flat_idx, next_split] == 0` at trajectory boundary (backpressure)
- ZMQ message: `(flat_idx, op)` = **8 bytes**
- All `infer_*` access: `self._infer_obs[flat_idx]` (1D, int Python scalar)

### InferenceServer
- Parse requests as Nx2 int32 array `[flat_idx, op]`
- Pre-cache numpy views in `serve()`: `self._obs_np = self.infer_obs.numpy()` etc.
- `_gather_obs`: `self._obs_np[flat_idxs]` — numpy 1D int32 fancy indexing, no `astype`, `torch.from_numpy` wraps zero-copy
- All scatter: `self._act_np[flat_idxs] = actions.numpy()` — numpy int32 scatter

### Learner
- Replace `traj_buffer_queue.put(ti)` → `split_flags[flat_idx, split_idx] = 0`

**Profiling**:

| Phase | Plan B | Plan C | Change |
|-------|--------|--------|--------|
| gather_obs | 0.42ms | 0.41ms | −2% |
| scatter | 0.20ms | 0.17ms | −15% |
| SPS | 8,600 | 9,700 | +13% |

gather_obs barely moved: the bottleneck is the shared memory read itself (cross-core cache coherency fetching Worker-written cache lines), not Python dispatch. This is the floor for the current architecture.

SPS gain (+13%) comes primarily from scatter improvement and removal of `mp.Queue` coordination overhead at trajectory boundaries.

**Distance to Sample Factory**: 9,700 vs 10,322 SPS = 6% gap. Remaining gap is likely ZMQ overhead (vs SF's in-process Python Queue) and IS process boundary (vs SF's in-process inference thread).

---

## Architecture Invariant

The following remains unchanged across all versions:

- Obs is written to `traj_tensors["obs"][ti, s]` by the Worker for training (not by IS)
- IS never touches `traj_tensors` directly
- Results (action, logp, value) are written by IS to infer_* buffers, then copied to `traj_tensors` by Workers
- LSTM live state capture timing: Worker captures input rnn_state in `_send_request` BEFORE sending ZMQ message (IS overwrites live buffer with output state)
- `ready_flags` remains the sole IS→Worker completion signal (no ZMQ reply)

## Current Message Format

```
Worker → IS:  struct.pack("<ii", flat_idx, op)   # 8 bytes
Worker → DS:  struct.pack("<i",  traj_idx)        # 4 bytes
DS → Learner: struct.pack("<i",  traj_idx)        # 4 bytes (forwarded)
```

## gather_obs Floor Analysis

For N=128 envs, obs_dim=44 (Humanoid):
- Data volume: 128 × 44 × 4 bytes = 22.5 KB per IS call
- Cache coherency: each obs was written by a different Worker core → cache line fetch required
- Measured floor: ~0.4ms → ~56 GB/s effective bandwidth (within DRAM range, not in-cache)

To go below this floor would require either:
1. Obs in ZMQ message (eliminates shared memory read; impractical for large obs like images)
2. Binding Worker and IS to same NUMA node / LLC domain
3. Reducing obs dimension
