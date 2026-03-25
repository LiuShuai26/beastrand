# projects/tennis/config.py
"""Configuration for Tennis self-play training.

Combines self-play mechanics (opponent IS, snapshot pool) with Atari-style
hyperparameters (NatureCNN policy, frame stacking, linear LR decay).
"""
from dataclasses import dataclass, field
from typing import Optional, List
import time


@dataclass
class Args:
    # --- Module paths ---
    data_record_path: str = field(default="beastrand.ppo.data_record.PPODataRecord", init=False)
    policy_path: str = field(default="projects.atari.policy.AtariPolicy", init=False)
    algorithm_path: str = field(default="beastrand.ppo.algorithm.PPOAlgorithm", init=False)
    make_env_path: str = field(default="projects.tennis.make_env.make_env", init=False)

    bootstrap_value: bool = field(default=True, init=False)

    # --- Multi-agent / Self-play ---
    num_agents: int = field(default=2, metadata={"help": "Agents per env (2 for Tennis)"})
    opp_refresh_interval: int = field(default=100, metadata={"help": "IS batches between opponent snapshot refresh"})
    snapshot_save_interval: int = field(default=10, metadata={"help": "Policy versions between snapshot saves"})
    max_snapshots: int = field(default=20, metadata={"help": "Max historical snapshots in pool"})

    # --- Devices ---
    inference_device: str = field(default="cpu", metadata={"help": "InferenceServer device (cpu/cuda/mps)"})
    learner_device: str = field(default="cpu", metadata={"help": "Learner device (cpu/cuda/mps)"})
    num_inference_servers: int = field(default=1, metadata={"help": "Number of InferenceServer processes (for agent 0)"})

    # --- Environment ---
    env_id: str = field(default="tennis_v3", metadata={"help": "Env identifier (used for run naming)"})
    seed: int = field(default=1, metadata={"help": "Random seed"})
    total_env_steps: int = field(default=10_000_000, metadata={"help": "Total environment steps"})
    max_policy_lag: int = field(default=0, metadata={"help": "Discard trajectories with policy lag > this (0 = disabled)"})

    # --- Topology ---
    num_workers: int = field(default=8, metadata={"help": "Number of rollout worker processes"})
    num_envs_per_worker: int = field(default=1, metadata={"help": "Environments per worker"})
    rollout: int = field(default=128, metadata={"help": "Unroll horizon"})

    # --- Atari-specific ---
    frame_stack: int = field(default=4, metadata={"help": "Number of stacked frames"})

    # --- RL / PPO (Atari-tuned, SF-matched) ---
    gamma: float = field(default=0.99, metadata={"help": "Discount factor"})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda"})
    replay_capacity: int = field(default=1024, metadata={"help": "Max transitions stored"})
    learning_starts: int = field(default=1024, metadata={"help": "Steps before first update"})
    batch_size: int = field(default=1024, metadata={"help": "Learner batch size"})
    minibatch_size: int = field(default=256, metadata={"help": "Learner minibatch size"})
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Learning rate"})
    lr_schedule: str = field(default="linear_decay", metadata={"help": "LR schedule"})
    train_epochs: int = field(default=4, metadata={"help": "PPO epochs per batch"})
    ppo_clip_range: float = field(default=0.1, metadata={"help": "PPO clip range"})
    ppo_clip_value: float = field(default=1.0, metadata={"help": "PPO value clip range"})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm"})
    entropy_coef: float = field(default=0.01, metadata={"help": "Entropy coefficient"})
    value_coef: float = field(default=0.5, metadata={"help": "Value loss coefficient"})
    normalize_adv: bool = field(default=True, metadata={"help": "Normalize advantages"})
    normalize_returns: bool = field(default=True, metadata={"help": "Normalize value targets"})
    normalize_input: bool = field(default=True, metadata={"help": "Normalize observations"})
    kl_loss_coeff: float = field(default=0.0, metadata={"help": "KL penalty coefficient (0=disabled)"})
    sync_training: bool = field(default=False, metadata={"help": "Sync mode"})

    # --- Logging / checkpoints ---
    logdir: str = field(default="train_logs", metadata={"help": "Root log directory"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Override run name"})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "W&B project name"})
    eval_interval: int = field(default=0, metadata={"help": "Eval every N env steps (0 = disabled)"})
    num_eval_episodes: int = field(default=10, metadata={"help": "Eval episodes"})
    checkpoint_interval: int = field(default=0, metadata={"help": "Checkpoint every N env steps"})
    resume: Optional[str] = field(default=None, metadata={"help": "Resume from checkpoint"})
    max_checkpoints: int = field(default=5, metadata={"help": "Max checkpoints to keep"})

    def make_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"tennis_selfplay_{int(time.time())}"

    def validate(self) -> None:
        assert self.num_workers > 0
        assert self.rollout > 0
        assert self.total_env_steps > 0
        assert self.num_envs_per_worker > 0
        assert self.num_agents == 2, "Tennis is a 2-player game"

    def __post_init__(self):
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("batch_size must be a multiple of minibatch_size")
        if self.batch_size % self.rollout != 0:
            raise ValueError("batch_size must be a multiple of rollout")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least as large as batch_size")
        if self.replay_capacity % self.batch_size != 0:
            raise ValueError("replay_capacity must be a multiple of batch_size")
        if self.learning_starts < self.replay_capacity:
            raise ValueError("learning_starts must be at least as large as replay_capacity")
        if self.learning_starts % self.batch_size != 0:
            raise ValueError("learning_starts must be at least a multiple of batch_size")
        self.validate()
