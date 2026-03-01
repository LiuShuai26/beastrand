# run/run_ppo_amp/ppo_amp_config.py
"""
Configuration for PPO-AMP training.

Extends the PPO config with AMP-specific fields: discriminator hyperparameters,
reward group weights, AMP observation slices, and keyframe file path.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class Args:
    # --- Module paths (PPO-AMP variants) ---
    data_record_path: str = "modules.dataset.data_record.ppo_amp_data_record.PPOAMPDataRecord"
    policy_path: str = "modules.policy.ppo_amp_policy.PPOAMPPolicy"
    algorithm_path: str = "modules.algos.ppo_amp.PPOAMPAlgorithm"
    make_env_path: str = "modules.envs.make_env_amp.make_env_amp"

    buffer_mode: str = "fullpass"
    compute_gae: bool = True
    bootstrap_value: bool = True

    device: str = field(default="cpu", metadata={"help": "Device to run on (cpu/cuda)"})

    # --- Environment ---
    env_id: str = field(default="Humanoid-v5", metadata={"help": "Gym env id"})
    seed: int = field(default=1, metadata={"help": "Random seed"})
    total_env_steps: int = field(default=50_000_000, metadata={"help": "Total environment steps"})

    # --- Beast .so env config ---
    brain_class: str = field(default="VAEBrain", metadata={"help": "Brain class: Brain or VAEBrain"})
    reward_mode: str = field(default="idle", metadata={"help": "Reward mode: idle, walk, or punch"})
    target_vx: float = field(default=1.5, metadata={"help": "Target forward velocity for walk mode"})

    lag_controll: bool = field(default=False, metadata={"help": "Lag control flag"})
    policy_lag: int = field(default=1, metadata={"help": "Max policy lag"})

    # --- Topology ---
    num_workers: int = field(default=8, metadata={"help": "Number of rollout worker processes"})
    num_envs_per_worker: int = field(default=2, metadata={"help": "Environments per worker"})
    rollout: int = field(default=512, metadata={"help": "Unroll horizon (steps per trajectory)"})

    # --- Policy ---
    mlp_layers: List[int] = field(default_factory=lambda: [256, 256], metadata={"help": "MLP hidden layers"})

    # --- RL / PPO ---
    gamma: float = field(default=0.95, metadata={"help": "Discount factor"})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda"})
    replay_capacity: int = field(default=8192, metadata={"help": "Max transitions stored (learner buffer size)"})
    learning_starts: int = field(default=8192, metadata={"help": "Steps before first update"})
    batch_size: int = field(default=8192, metadata={"help": "Learner batch size"})
    minibatch_size: int = field(default=512, metadata={"help": "Learner minibatch size"})
    num_minibatches: int = field(default=16, metadata={"help": "Number of minibatches (for disc)"})
    learning_rate: float = field(default=1e-4, metadata={"help": "Policy learning rate"})
    entropy_coef: float = field(default=0.001, metadata={"help": "Entropy coefficient"})
    value_coef: float = field(default=0.5, metadata={"help": "Value loss coefficient"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    train_epochs: int = field(default=2, metadata={"help": "PPO epochs per batch"})
    ppo_clip_range: float = field(default=0.2, metadata={"help": "PPO clip range"})
    ppo_clip_value: float = field(default=1.0, metadata={"help": "PPO value clip range"})
    normalize_adv: bool = field(default=True, metadata={"help": "Normalize advantages"})

    # --- AMP-specific ---
    keyframe_file: str = field(default="", metadata={"help": "Path to keyframe JSON file (required for AMP)"})

    amp_obs_slices: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 1), (6, 30), (42, 52), (76, 77)],
        metadata={"help": "Observation index slices for AMP features [(start, end), ...]"},
    )
    amp_obs_dim: int = field(default=0, metadata={"help": "AMP obs dimension (computed from slices)"})

    task_reward_weight: float = field(default=0.1, metadata={"help": "Weight for task reward group"})
    style_reward_weight: float = field(default=0.9, metadata={"help": "Weight for style reward group"})

    disc_lr: float = field(default=5e-4, metadata={"help": "Discriminator learning rate"})
    disc_hidden_dim: int = field(default=256, metadata={"help": "Discriminator hidden layer size"})
    disc_num_layers: int = field(default=2, metadata={"help": "Number of discriminator hidden layers"})
    disc_grad_penalty_coef: float = field(default=5.0, metadata={"help": "Gradient penalty coefficient"})
    disc_weight_decay: float = field(default=1e-4, metadata={"help": "Discriminator weight decay"})
    disc_update_epochs: int = field(default=1, metadata={"help": "Discriminator epochs per PPO iteration"})
    disc_noise_std: float = field(default=0.0, metadata={"help": "Gaussian noise std on disc inputs during training (0=off)"})

    # --- Logging / checkpoints ---
    logdir: str = field(default="train_logs", metadata={"help": "Root log directory"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Override run name"})
    eval_interval: int = field(default=10_000, metadata={"help": "Eval every N env steps"})
    checkpoint_interval: int = field(default=5_000_000, metadata={"help": "Save checkpoint every N env steps (0 = only on exit)"})

    # ---- helpers ----
    def make_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = int(time.time())
        return f"{self.env_id}_amp_{ts}"

    def validate(self) -> None:
        assert self.num_workers > 0
        assert self.rollout > 0
        assert self.total_env_steps > 0
        assert self.num_envs_per_worker > 0
        assert self.keyframe_file, "--keyframe-file is required for AMP training"

    def __post_init__(self):
        # Compute amp_obs_dim from slices
        if self.amp_obs_dim == 0:
            self.amp_obs_dim = sum(e - s for s, e in self.amp_obs_slices)

        # Validation
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
