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
