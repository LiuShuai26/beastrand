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

### Implemented Algorithms
* PPO (Proximal Policy Optimization)

* SAC (Soft Actor-Critic)

Algorithm code is referenced from CleanRL
