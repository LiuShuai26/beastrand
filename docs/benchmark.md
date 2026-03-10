# Benchmark: Humanoid-v4 PPO (beastrand vs Sample Factory)

**Date**: 2026-03-09
**Environment**: Humanoid-v4, 10M steps
**Test machine**: AMD Ryzen 9 5950X (16c/32t) · 32 GB RAM · NVIDIA RTX 5070 Ti 16 GB
**Topology**: 8 workers × 8 envs = 64 envs total

> **Note**: The goal of this benchmark is to show that beastrand and Sample Factory operate at the
> same level of throughput and training efficiency — not to claim superiority. Reward results vary
> significantly across seeds, and each framework has its own hyperparameter sweet spot that differs
> per algorithm and environment. Fair apples-to-apples tuning would require separate hyperparameter
> searches for each framework.

## Results

| Framework | FPS | Reward (seed 42) | Reward (seed 123) | Reward (seed 7) | Reward (mean) |
|-----------|-----|------------------|-------------------|-----------------|---------------|
| **beastrand** | **17,442** | 3,582 | 6,901 | 6,735 | **5,739** |
| Sample Factory | 14,873 | 6,814 | 6,218 | 4,906 | 5,979 |

### Per-Seed FPS

| Seed | beastrand FPS | SF FPS |
|------|--------------|--------|
| 42   | 18,022       | 15,128 |
| 123  | 17,382       | 15,155 |
| 7    | 16,921       | 14,336 |
| mean | **17,442**   | 14,873 |

## beastrand Command

```bash
python -m projects.mujoco.train \
  --env-id Humanoid-v4 \
  --seed 42 \
  --total-env-steps 10000000
```

All other hyperparameters use the `projects/mujoco/config.py` defaults (SF-matched).

## Sample Factory Command

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
