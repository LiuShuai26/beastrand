# Beastrand

A distributed reinforcement learning framework for training physics-based game AI.

As simple as [CleanRL](https://github.com/vwxyzjn/cleanrl), as fast as [Sample Factory](https://github.com/alex-petrenko/sample-factory).

## Features

- **High throughput** — multi-process architecture with shared-memory tensors and ZMQ IPC, zero pickle on the hot path
- **Simple & readable** — each component is a short, self-contained script; no deep abstractions
- **Modular** — each algorithm is a self-contained entry point; easy to fork and customize
- **Flexible environments** — works with any Gymnasium env or custom C++ environments compiled as `.so` modules

### Algorithms

- PPO (Proximal Policy Optimization)
- PPO-LSTM (PPO with recurrent policy)
- PPO-AMP (Adversarial Motion Priors for motion imitation)

## Installation

Requires Python 3.10+ and PyTorch 2.0+.

```bash
pip install -r requirements.txt
```

## Usage

### PPO

```bash
# Gymnasium environment
python -m run.run_ppo.train_ppo --env-id Humanoid-v5

# Customize training
python -m run.run_ppo.train_ppo --env-id CartPole-v1 --num-workers 4 --device cuda

# LSTM policy
python -m run.run_ppo.train_ppo --env-id Humanoid-v5 --use-lstm True
```

### PPO-AMP

```bash
# Beast .so environment with adversarial motion priors
python -m run.run_ppo_amp.train_ppo_amp \
  --env-id HumanoidEnv \
  --keyframe-file path/to/keyframes.json \
  --reward-mode walk \
  --target-vx 1.5
```

### Monitoring

```bash
tensorboard --logdir train_logs/
```

All config fields are CLI flags via [tyro](https://github.com/brentyi/tyro). Run `--help` to see the full list.

## Adding a New Algorithm

Like CleanRL, each algorithm is a self-contained entry point. To add your own:

1. Copy `run/run_ppo/` to `run/run_my_algo/`
2. Implement three modules:
   - **Policy** — `forward()` and `get_action()` (see `modules/policy/ppo_policy.py`)
   - **Algorithm** — `train_step(policy, batch)` (see `modules/algos/ppo.py`)
   - **DataRecord** — trajectory buffer fields (see `modules/dataset/data_record/ppo_data_record.py`)
3. Point your config to the new modules and run

The distributed infrastructure (workers, inference server, learner) is reused — you only write the algorithm-specific parts.

### Custom Environments

Any Gymnasium-compatible environment works out of the box. For custom env factories:

```bash
python -m run.run_ppo.train_ppo --env-id MyEnv --make-env-path my_project.envs.make_my_env
```
