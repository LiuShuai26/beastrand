# beastrand

A high-throughput distributed reinforcement learning framework designed to be read, understood, and modified.

As simple as [CleanRL](https://github.com/vwxyzjn/cleanrl), as fast as [Sample Factory](https://github.com/alex-petrenko/sample-factory).

## Why beastrand?

Existing frameworks force a choice: **simple but slow** (CleanRL, SB3) or **fast but opaque** (Sample Factory, RLlib). beastrand refuses this trade-off. It delivers async multi-process throughput in ~3K lines of core code where every component is a short, self-contained file you can read and change without fighting the framework.

**Use beastrand if you:**
- Need high throughput without sacrificing readability
- Need custom observations, custom action spaces, custom model architectures, and custom training loops. beastrand stays out of your way on all four
- Need a highly customizable distributed architecture. beastrand's clean multi-process design serves as a solid foundation to extend to multi-machine or custom distributed setups

**Don't use beastrand if you:**
- Want many algorithms out of the box (DQN, SAC, TD3, etc.) → use [SB3](https://github.com/DLR-RM/stable-baselines3), [Tianshou](https://github.com/thu-ml/tianshou), or [RLlib](https://docs.ray.io/en/latest/rllib/)
- Need single-machine peak PPO performance with zero setup → use [Sample Factory](https://github.com/alex-petrenko/sample-factory)
- Need multi-machine distributed training → use [RLlib](https://docs.ray.io/en/latest/rllib/)
- Want the simplest possible single-file baseline → use [CleanRL](https://github.com/vwxyzjn/cleanrl)

## Features

- **High throughput** — multi-process architecture with shared-memory tensors and ZMQ IPC; zero pickle on the hot path; async collection with batched GPU inference
- **Simple & readable** — ~3K lines of core code; each component is a short, self-contained file; no deep abstractions
- **Modular** — policy, algorithm, data record, and env factory are swappable via dotted Python paths; adding a new algorithm variant means adding one directory with 5 files, zero changes to core
- **Flexible environments** — works with any Gymnasium env or custom C++ environments compiled as `.so` modules

## Installation

Requires Python 3.10+ and PyTorch 2.0+.

```bash
pip install -e .
```

## Usage

```bash
# Standard PPO
python -m beastrand.ppo.train --env-id Humanoid-v5

# PPO-LSTM
python -m projects.ppo_lstm.train --env-id Humanoid-v5

# PPO-AMP (motion imitation with Beast .so environment)
python -m projects.ppo_amp.train --env-id HumanoidEnv \
  --keyframe-file path/to/keyframes.json

# MuJoCo preset (SF-tuned hyperparams)
python -m projects.mujoco.train --env-id Humanoid-v4

# Atari (CNN policy, frame stacking)
python -m projects.atari.train --env-id BreakoutNoFrameskip-v4

# Monitoring
tensorboard --logdir train_logs/
```

All config fields are CLI flags via [tyro](https://github.com/brentyi/tyro). Run `--help` to see the full list.

## Benchmarks

### Humanoid-v4 — 10M steps (seeds 42, 123, 7)

**Test machine:** AMD Ryzen 9 5950X (16c/32t) · 32 GB RAM · NVIDIA RTX 5070 Ti 16 GB · 8 workers × 8 envs = 64 envs total

| Framework      | FPS        | Reward (seed 42) | Reward (seed 123) | Reward (seed 7) | Reward (mean) |
|----------------|------------|------------------|-------------------|-----------------|---------------|
| **beastrand**  | 16,226     | 5,858            | 5,392             | 5,535           | 5,595         |
| Sample Factory | **18,438** | 6,274            | 5,979             | 6,184           | **6,146**     |

| Seed | beastrand FPS | SF FPS     |
|------|---------------|------------|
| 42   | 16,534        | 18,639     |
| 123  | 15,571        | 18,517     |
| 7    | 16,573        | 18,159     |
| mean | 16,226        | **18,438** |

Both frameworks achieve comparable throughput and training efficiency. Reward results vary across seeds; each framework has its own hyperparameter sweet spot per algorithm and environment. See [docs/benchmark.md](docs/benchmark.md) for hyperparameters and methodology.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full deep dive into the multi-process architecture, shared-memory data flow, and module system.

## Projects

Extensions built on top of the core framework:

| Project | What it adds |
|---------|-------------|
| `projects/ppo_lstm` | Recurrent policy with truncated BPTT |
| `projects/ppo_amp` | Adversarial motion priors for physics-based character animation |
| `projects/mujoco` | MuJoCo-tuned hyperparameters |
| `projects/atari` | CNN policy with frame stacking |

Each project is a self-contained directory (policy, algorithm, data record, config, train entry point). All were added with **zero changes to core code**.