# Benchmark: Humanoid-v4 PPO (beastrand vs Sample Factory)

**Date**: 2026-03-08
**Environment**: Humanoid-v4, 10M steps, CPU

## Results

| Framework | Seeds | Peak (mean) | Final 10M (mean) |
|-----------|-------|-------------|-------------------|
| **beastrand** | 42, 123, 7 | **5761 +/- 181** | **5735 +/- 168** |
| Sample Factory (async) | 1 | 6095 | 6042 |

**Gap: ~5%** (within 1 sigma of SF)

### Per-Seed Breakdown (beastrand)

| Seed | Peak | Final (10M) | Policy Versions |
|------|------|-------------|-----------------|
| 42   | 5554 | 5540        | 2442            |
| 123  | 5899 | 5833        | 2444            |
| 7    | 5831 | 5831        | 2441            |

### Learning Curve Milestones (seed 42)

| Steps | Reward | Avg Episode Length |
|-------|--------|--------------------|
| 1M    | 539    | 104                |
| 2M    | 727    | 139                |
| 3M    | 1049   | 195                |
| 4M    | 1597   | 294                |
| 5M    | 2752   | 477                |
| 6M    | 4385   | 752                |
| 7M    | 4698   | 797                |
| 8M    | 5204   | 859                |
| 9M    | 5764   | 936                |
| 10M   | 5540   | 912                |

## Command

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

## Sample Factory Reference Run

```bash
python -m sample_factory.huggingface.load_from_hub -r edbeeching/ppo_mujoco_humanoid-v4 -d ./sf_train_dir

python -m sf_examples.mujoco.train_mujoco \
  --env=mujoco_humanoid \
  --train_dir=./sf_train_dir \
  --experiment=Humanoid-v4 \
  --device=cpu \
  --async_rl=True \
  --train_for_env_steps=10000000
```

SF async result: peak 6095, final 6042, policy lag avg=10, max=15.
