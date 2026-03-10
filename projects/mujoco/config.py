# projects/mujoco/config.py
"""MuJoCo-tuned PPO configuration (SF-matched hyperparameters)."""

from dataclasses import dataclass, field

from beastrand.ppo.config import Args as PPOArgs


@dataclass
class Args(PPOArgs):

    # ---- MuJoCo-tuned defaults ----
    env_id: str = field(default="HalfCheetah-v4", metadata={"help": "MuJoCo env id"})
    total_env_steps: int = field(default=10_000_000, metadata={"help": "Total environment steps"})

    # Topology
    num_workers: int = field(default=8, metadata={"help": "Number of rollout worker processes"})
    num_envs_per_worker: int = field(default=8, metadata={"help": "Environments per worker"})
    rollout: int = field(default=64, metadata={"help": "Unroll horizon"})

    # Policy
    mlp_layers: list[int] = field(default_factory=lambda: [64, 64], metadata={"help": "MLP layers"})

    # RL
    batch_size: int = field(default=4096, metadata={"help": "Learner batch size"})
    minibatch_size: int = field(default=1024, metadata={"help": "Learner minibatch size"})
    replay_capacity: int = field(default=4096, metadata={"help": "Learner buffer size"})
    learning_starts: int = field(default=4096, metadata={"help": "Collect-only steps before training"})
    train_epochs: int = field(default=2, metadata={"help": "Training epochs per batch"})
    learning_rate: float = field(default=2.95e-3, metadata={"help": "Learning rate"})
    lr_schedule: str = field(default="linear_decay", metadata={"help": "LR schedule"})
    ppo_clip_range: float = field(default=0.2, metadata={"help": "PPO clip range"})
    max_grad_norm: float = field(default=3.5, metadata={"help": "Max gradient norm"})
    value_coef: float = field(default=1.3, metadata={"help": "Value loss coefficient"})
    entropy_coef: float = field(default=0.0, metadata={"help": "Entropy coefficient"})
    kl_loss_coeff: float = field(default=0.1, metadata={"help": "KL penalty coefficient"})
    normalize_input: bool = field(default=True, metadata={"help": "Normalize observations"})
    normalize_returns: bool = field(default=True, metadata={"help": "Normalize value targets"})
