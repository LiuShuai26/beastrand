# projects/atari/config.py
"""Atari PPO configuration (SF-matched hyperparameters)."""

from dataclasses import dataclass, field

from beastrand.ppo.config import Args as PPOArgs


@dataclass
class Args(PPOArgs):

    policy_path: str = field(default="projects.atari.policy.AtariPolicy", init=False)
    make_env_path: str = field(default="projects.atari.make_env.make_env", init=False)

    # ---- Atari-tuned defaults ----
    env_id: str = field(default="BreakoutNoFrameskip-v4", metadata={"help": "Atari env id"})
    total_env_steps: int = field(default=10_000_000, metadata={"help": "Total environment steps"})

    # Topology
    num_workers: int = field(default=8, metadata={"help": "Number of rollout worker processes"})
    num_envs_per_worker: int = field(default=1, metadata={"help": "Environments per worker"})
    rollout: int = field(default=128, metadata={"help": "Unroll horizon"})

    # RL (CleanRL Atari defaults)
    batch_size: int = field(default=1024, metadata={"help": "Learner batch size"})
    minibatch_size: int = field(default=256, metadata={"help": "Learner minibatch size"})
    replay_capacity: int = field(default=1024, metadata={"help": "Learner buffer size"})
    learning_starts: int = field(default=1024, metadata={"help": "Collect-only steps before training"})
    train_epochs: int = field(default=4, metadata={"help": "Training epochs per batch"})
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Learning rate"})
    lr_schedule: str = field(default="linear_decay", metadata={"help": "LR schedule"})
    ppo_clip_range: float = field(default=0.1, metadata={"help": "PPO clip range"})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm"})
    value_coef: float = field(default=0.5, metadata={"help": "Value loss coefficient"})
    entropy_coef: float = field(default=0.01, metadata={"help": "Entropy coefficient"})
    kl_loss_coeff: float = field(default=0.0, metadata={"help": "KL penalty coefficient"})
    # Atari obs are pixel images; /255 in AtariPolicy._preprocess is the canonical
    # normalization. Per-pixel running mean/std (RunningMeanStdTorch over 4*84*84)
    # collapses spatial structure for the CNN and breaks training. Default off,
    # matches CleanRL / Sample Factory practice.
    normalize_input: bool = field(default=False, metadata={"help": "Per-pixel running mean/std on images (off — use /255 only)"})
    normalize_returns: bool = field(default=True, metadata={"help": "Normalize value targets (SF-style)"})

    # Atari-specific
    frame_stack: int = field(default=4, metadata={"help": "Number of stacked frames"})
