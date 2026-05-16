# Benchmark: Humanoid-v4 PPO (beastrand vs Sample Factory)

**Date**: 2026-03-10
**Environment**: Humanoid-v4, 10M steps
**Test machine**: AMD Ryzen 9 5950X (16c/32t) · 32 GB RAM · NVIDIA RTX 5070 Ti 16 GB

> **Note**: The goal of this benchmark is to show that beastrand and Sample Factory operate at the
> same level of throughput and training efficiency. Reward results vary
> across seeds, and each framework has its own hyperparameter sweet spot that differs
> per algorithm and environment.

## Results (8w × 8e = 64 envs)

| Framework      | FPS        | Reward (seed 42) | Reward (seed 123) | Reward (seed 7) | Reward (mean) |
|----------------|------------|------------------|-------------------|-----------------|---------------|
| **beastrand**  | 16,226     | 5,858            | 5,392             | 5,535           | 5,595         |
| Sample Factory | **18,438** | 6,274            | 5,979             | 6,184           | **6,146**     |

### Per-Seed FPS (8w × 8e)

| Seed | beastrand FPS | SF FPS     |
|------|---------------|------------|
| 42   | 16,534        | 18,639     |
| 123  | 15,571        | 18,517     |
| 7    | 16,573        | 18,159     |
| mean | 16,226        | **18,438** |

## Scaling: 24w × 8e = 192 envs (beastrand only)

| Metric | seed 42 | seed 123 | seed 7 | mean       |
|--------|---------|----------|--------|------------|
| Reward | 5,829   | 5,704    | 5,868  | 5,800      |
| FPS    | 34,948  | 34,690   | 34,315 | **34,651** |

| Resource    | Utilization           |
|-------------|-----------------------|
| GPU compute | 48%                   |
| GPU memory  | 788 MiB / 16,303 MiB (5%) |
| CPU usr     | 81%                   |
| CPU idle    | 17%                   |

## Commands

### beastrand

```bash
# 8w × 8e (default)
python -m projects.mujoco.train \
  --env-id Humanoid-v4 \
  --seed 42 \
  --total-env-steps 10000000

# 24w × 8e
python -m projects.mujoco.train \
  --env-id Humanoid-v4 \
  --seed 42 \
  --total-env-steps 10000000 \
  --num-workers 24
```

All other hyperparameters use the `projects/mujoco/config.py` defaults (SF-matched).

### Sample Factory

```bash
python -m sf_examples.mujoco.train_mujoco \
  --env=mujoco_humanoid \
  --train_dir=./sf_train_dir \
  --experiment=Humanoid-v4 \
  --seed=42 \
  --async_rl=True \
  --train_for_env_steps=10000000
```

## Shared Hyperparameters

Both frameworks use identical hyperparameters (SF's MuJoCo defaults from `sf_examples/mujoco/mujoco_params.py`, matched in `projects/mujoco/config.py`):

| Parameter | Value |
|-----------|-------|
| num_workers | 8 |
| num_envs_per_worker | 8 |
| rollout | 64 |
| batch_size | 4096 (SF: 1024 minibatch × 4) |
| num_epochs | 2 |
| learning_rate | 0.00295 (linear decay) |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| ppo_clip_ratio | 0.2 |
| ppo_clip_value | 1.0 |
| max_grad_norm | 3.5 |
| value_loss_coeff | 1.3 |
| entropy_coeff | 0.0 |
| kl_loss_coeff | 0.1 |
| normalize_input | True |
| normalize_returns | True |
| mlp_layers | [64, 64], tanh |
| adaptive_stddev | False |
| shuffle_minibatches | False |

---

# PPO-LSTM Reference (Humanoid-v4)

**Date**: 2026-05-16 · **HEAD**: ff7100a (audit follow-up wave 3)
**Test machine**: same as above (Ryzen 9 5950X · 32 GB · RTX 5070 Ti).

First reference numbers for the recurrent variant. Run via
`bash scripts/benchmark_humanoid.sh lstm` (3 seeds × 10M steps).

## Results (8w × 8e = 64 envs)

| seed | duration | FPS    | reward (8-worker mean) |
|------|---------:|-------:|-----------------------:|
| 42   |    714 s | 14,002 |                  5,387 |
| 123  |    713 s | 14,022 |                  6,594 |
| 7    |    690 s | 14,493 |                  6,773 |
| mean |        — | **14,172** |              **6,251** |

## LSTM vs PPO (same machine, same code, same seeds)

| Metric          | PPO    | PPO-LSTM | Δ        |
|-----------------|-------:|---------:|---------:|
| FPS (mean)      | 15,968 |   14,172 | **-11.2%** |
| Reward (mean)   |  5,775 |    6,251 | **+8.2%**  |

LSTM throughput tax (~11%) comes from BPTT + RNN-state propagation; the
mean reward is higher but **seed-dependent**: LSTM under-performs PPO on
seed 42 (5,387 vs 6,162) while substantially out-performing on seeds 123
and 7. Humanoid-v4 is a fully observable MDP, so the lift is not a
free-lunch — likely some combination of temporal smoothing and momentum
effects. Three seeds is too small for a statistical claim; treat the
+8.2% mean as illustrative, not nominal.

## Command

```bash
python -m projects.ppo_lstm.train \
  --env-id Humanoid-v4 \
  --seed 42 \
  --total-env-steps 10000000 \
  --run-name bench_lstm_s42
```

Hyperparameters are the SF-matched mujoco defaults plus LSTM-specific
fields (`lstm_hidden_size`, `recurrence`, etc.) from
`projects/ppo_lstm/config.py`. The numerical-safety ratio clamp
(`compute_clamped_ratio`, [0.05, 20.0]) added in wave 3 is active on this
run.
