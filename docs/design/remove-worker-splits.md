# Decision: Remove worker_num_splits (Double Buffering)

**Date**: 2026-03-07
**Status**: Accepted

## Context

RolloutWorker had a `worker_num_splits` parameter that split envs into N groups
for "double-buffered" pipelined inference. The idea was to burst-send inference
requests per group so the InferenceServer could form larger GPU batches.

## Analysis

### Per-env async polling is already optimal for CPU envs

With `worker_num_splits=0` (per-env mode), each env independently cycles:

```
send inference request → poll ready_flag → step → send next request
```

The worker loop polls all envs every iteration. Whichever env's inference result
arrives first gets stepped immediately. CPU is never idle waiting — while one env
waits for GPU, others can step. This is a **finer-grained pipeline** than any
N-way split can achieve.

### Split mode introduces unnecessary synchronization

With `worker_num_splits=2`, envs are grouped. All envs in a group must be
stepped before their requests are burst-sent together. This **delays** sending
requests for envs that finish early, adding latency without benefit.

### GPU batching already works across workers

The InferenceServer collects requests from **all workers** (e.g., 8 workers ×
8 envs = 64 envs). It naturally forms large batches regardless of whether
individual workers send requests one-at-a-time or in bursts.

### Benchmark confirmation

```
| Config           | SPS    | GPU | CPU usr | CPU idle | forward | batch_size |
| ---------------- | ------ | --- | ------- | -------- | ------- | ---------- |
| 24w×8e split=1   | 27,063 | 65% | 81%     | 16%      | 0.82ms  | ~44,500    |
| 24w×8e split=2   | 27,027 | 60% | 89%     | 9%       | 0.79ms  | ~43,000    |
```

SPS is identical, but split=2 uses **more CPU** (89% vs 81%) and **less GPU**
(60% vs 65%) due to coordination overhead. Per-env mode (split=1/0) is strictly
better.

### When double buffering IS useful

Double buffering helps when env.step() and inference are **both on GPU** (e.g.,
Isaac Gym / IsaacLab). In that case, step and inference compete for the same
GPU, and splitting into groups on different CUDA streams allows them to overlap.

In beastrand, env.step() runs on **CPU** and inference runs on **GPU** — they
naturally run in parallel on different hardware. No double buffering needed.

## Decision

Remove `worker_num_splits` from:
- `ppo_config.py` — field and validation
- `ppo_amp_config.py` — field and validation
- `rollout_worker.py` — split logic, `_loop_split()` method, split initialization

Keep only the per-env async loop (`_loop_per_env`).

## Files Changed

- `nodes/rollout_worker.py` — removed `_loop_split()`, split init, simplified `run()`
- `run/run_ppo/ppo_config.py` — removed `worker_num_splits` field and validation
- `run/run_ppo_amp/ppo_amp_config.py` — removed `worker_num_splits` field and validation
