# Beastrand
Beastrand is a distributed reinforcement learning framework designed for high-throughput training on a single machine or across multiple processes.
It draws inspiration from IMPALA, TorchBeast, and Sample Factory, but is built with an emphasis on simplicity, modularity, and readability.

The goal of Beastrand is to be as simple as CleanRL and as high-throughput as Sample Factory.

### Key Ideas
* Easy to Understand & Modify
  
  Code is written to be readable first, not over-engineered.
  Each component is kept short and direct, using inheritance only where it improves clarity.
* Modular Node-Based Design
  
  The system is composed of nodes, each responsible for a single role.
  Every node has its own script, making it easy to run, inspect, or replace independently.

### Supported Environments
* **Gymnasium** — any standard Gym/Gymnasium environment (e.g. `Humanoid-v5`, `CartPole-v1`)
* **Beast .so** — custom compiled C++ environments (e.g. `HumanoidEnv`), loaded via `beastlab.env_loader`

The environment backend is auto-detected: the AMP factory tries Beast first and falls back to `gym.make()`. A custom factory can be specified via `--make-env-path`.

### Implemented Algorithms
* PPO (Proximal Policy Optimization)
* PPO-AMP (Adversarial Motion Priors)

Algorithm code is referenced from CleanRL
