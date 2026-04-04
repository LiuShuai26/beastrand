# projects/ppo_selfplay/config.py
"""Configuration for PPO self-play training.

Extends PPO config with multi-agent fields. Uses standard PPO policy/algorithm;
the self-play mechanics (opponent IS, snapshot pool) are handled by the framework
when num_agents > 1.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import time


@dataclass
class Args:
    # --- Module paths (standard PPO) ---
    data_record_path: str = field(default="beastrand.ppo.data_record.PPODataRecord", init=False)
    policy_path: str = field(default="beastrand.ppo.policy.PPOPolicy", init=False)
    algorithm_path: str = field(default="beastrand.ppo.algorithm.PPOAlgorithm", init=False)
    make_env_path: str = field(default="projects.ppo_selfplay.make_env.make_env_selfplay", init=False)

    bootstrap_value: bool = field(default=True, init=False)

    # --- Multi-agent / Self-play ---
    num_agents: int = field(default=2, metadata={"help": "Agents per env (1=single, 2=self-play)"})
    self_play_mode: str = field(
        default="mixed",
        metadata={"help": "Opponent mode: snapshot | latest | mixed"},
    )
    latest_self_play_ratio: float = field(
        default=0.7,
        metadata={"help": "When self_play_mode=mixed, probability of using latest-vs-latest for an episode"},
    )
    opp_refresh_interval: int = field(default=100, metadata={"help": "IS batches between opponent snapshot refresh"})
    snapshot_save_interval: int = field(default=10, metadata={"help": "Policy versions between snapshot saves"})
    max_snapshots: int = field(default=20, metadata={"help": "Max historical snapshots in pool"})

    # --- Devices ---
    inference_device: str = field(default="cpu", metadata={"help": "InferenceServer device (cpu/cuda/mps)"})
    learner_device: str = field(default="cpu", metadata={"help": "Learner device (cpu/cuda/mps)"})
    num_inference_servers: int = field(default=1, metadata={"help": "Number of InferenceServer processes (for agent 0)"})

    # --- Environment ---
    env_id: str = field(default="CartPole-v1", metadata={"help": "Gym env id"})
    seed: int = field(default=1, metadata={"help": "Random seed"})
    total_env_steps: int = field(default=500_000, metadata={"help": "Total environment steps"})
    max_policy_lag: int = field(default=0, metadata={"help": "Discard trajectories with policy lag > this (0 = disabled)"})

    # --- Topology ---
    num_workers: int = field(default=4, metadata={"help": "Number of rollout worker processes"})
    num_envs_per_worker: int = field(default=2, metadata={"help": "Environments per worker"})
    rollout: int = field(default=64, metadata={"help": "Unroll horizon"})

    # --- Policy ---
    mlp_layers: List[int] = field(default_factory=lambda: [256, 256], metadata={"help": "MLP layers"})

    # --- RL / PPO ---
    gamma: float = field(default=0.99, metadata={"help": "Discount factor"})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda"})
    replay_capacity: int = field(default=1024, metadata={"help": "Max transitions stored"})
    learning_starts: int = field(default=1024, metadata={"help": "Steps before first update"})
    batch_size: int = field(default=1024, metadata={"help": "Learner batch size"})
    minibatch_size: int = field(default=256, metadata={"help": "Learner minibatch size"})
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Learning rate"})
    entropy_coef: float = field(default=0.01, metadata={"help": "Entropy coefficient"})
    value_coef: float = field(default=1.0, metadata={"help": "Value loss coefficient"})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm"})
    train_epochs: int = field(default=1, metadata={"help": "PPO epochs per batch"})
    ppo_clip_range: float = field(default=0.2, metadata={"help": "PPO clip range"})
    ppo_clip_value: float = field(default=1.0, metadata={"help": "PPO value clip range"})
    normalize_adv: bool = field(default=True, metadata={"help": "Normalize advantages"})
    normalize_returns: bool = field(default=True, metadata={"help": "Normalize value targets"})
    normalize_input: bool = field(default=True, metadata={"help": "Normalize observations"})
    kl_loss_coeff: float = field(default=0.0, metadata={"help": "KL penalty coefficient (0=disabled)"})
    lr_schedule: str = field(default="constant", metadata={"help": "LR schedule: constant | linear_decay"})
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
        return f"{self.env_id}_selfplay_{int(time.time())}"

    def validate(self) -> None:
        assert self.num_workers > 0
        assert self.rollout > 0
        assert self.total_env_steps > 0
        assert self.num_envs_per_worker > 0
        assert self.num_agents >= 1
        if self.self_play_mode not in {"snapshot", "latest", "mixed"}:
            raise ValueError("self_play_mode must be 'snapshot', 'latest', or 'mixed'")
        if not (0.0 <= self.latest_self_play_ratio <= 1.0):
            raise ValueError("latest_self_play_ratio must be in [0, 1]")

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
            raise ValueError("learning_starts must be a multiple of batch_size")
        self.validate()
