# beastrand

A distributed reinforcement learning framework for training physics-based game AI.

As simple as [CleanRL](https://github.com/vwxyzjn/cleanrl), as fast as [Sample Factory](https://github.com/alex-petrenko/sample-factory).

## Features

- **High throughput** — multi-process architecture with shared-memory tensors and ZMQ IPC, zero pickle on the hot path
- **Simple & readable** — ~2K lines of core code; each component is a short, self-contained script; no deep abstractions
- **Modular** — policy, algorithm, data record, and env factory are swappable via dotted Python paths
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

Recommended: conda env `beastrand`.

## Usage

```bash
python -m ppo.train --env-id Humanoid-v5                # PPO
python -m projects.ppo_lstm.train --env-id Humanoid-v5   # PPO-LSTM
python -m projects.ppo_amp.train --env-id HumanoidEnv \  # PPO-AMP
  --keyframe-file path/to/keyframes.json

tensorboard --logdir train_logs/                          # Monitoring
```

All config fields are CLI flags via [tyro](https://github.com/brentyi/tyro). Run `--help` to see the full list.

See [docs/architecture.md](docs/architecture.md) for the full deep dive.
