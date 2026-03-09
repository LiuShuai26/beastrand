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

| Framework | FPS | seed 42 | seed 123 | seed 7 | mean |
|-----------|-----|---------|----------|--------|------|
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
python -m ppo.train \
  --env-id Humanoid-v4 \
  --seed 42 \
  --total-env-steps 10000000 \
  --num-workers 8 \
  --num-envs-per-worker 8 \
  --rollout 64 \
  --mlp-layers 64 64 \
  --batch-size 4096 \
  --minibatch-size 1024 \
  --train-epochs 2 \
  --replay-capacity 4096 \
  --learning-starts 4096 \
  --learning-rate 0.00295 \
  --lr-schedule linear_decay \
  --gamma 0.99 \
  --lam 0.95 \
  --ppo-clip-range 0.2 \
  --ppo-clip-value 1.0 \
  --max-grad-norm 3.5 \
  --value-coef 1.3 \
  --entropy-coef 0.0 \
  --kl-loss-coeff 0.1 \
  --normalize-input \
  --normalize-returns \
  --normalize-adv
```

### Full Resolved Args

```
Args(
  data_record_path='ppo.data_record.PPODataRecord',
  policy_path='ppo.policy.PPOPolicy',
  algorithm_path='ppo.algorithm.PPOAlgorithm',
  bootstrap_value=True,
  make_env_path=None,
  inference_device='cpu',
  learner_device='cpu',
  num_inference_servers=1,
  env_id='Humanoid-v4',
  seed=42,
  total_env_steps=10000000,
  max_policy_lag=0,
  num_workers=8,
  num_envs_per_worker=8,
  rollout=64,
  mlp_layers=[64, 64],
  gamma=0.99,
  lam=0.95,
  replay_capacity=4096,
  learning_starts=4096,
  batch_size=4096,
  minibatch_size=1024,
  learning_rate=0.00295,
  entropy_coef=0.0,
  value_coef=1.3,
  max_grad_norm=3.5,
  train_epochs=2,
  ppo_clip_range=0.2,
  ppo_clip_value=1.0,
  normalize_adv=True,
  normalize_returns=True,
  normalize_input=True,
  kl_loss_coeff=0.1,
  lr_schedule='linear_decay',
  logdir='train_logs',
  run_name=None,
  eval_interval=10000,
  checkpoint_interval=0,
)
```

## Sample Factory Command

```bash
python -m sf_examples.mujoco.train_mujoco \
  --env=mujoco_humanoid \
  --train_dir=./sf_train_dir \
  --experiment=Humanoid-v4 \
  --async_rl=True \
  --train_for_env_steps=10000000
```
