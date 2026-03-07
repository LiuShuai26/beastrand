# ppo_config.py
from dataclasses import dataclass, field
from typing import Optional, List
import time

@dataclass
class Args:

    data_record_path: str = field(default="modules.dataset.data_record.ppo_data_record.PPODataRecord", init=False)
    policy_path: str = field(default="modules.policy.ppo_policy.PPOPolicy", init=False)
    algorithm_path: str = field(default="modules.algos.ppo.PPOAlgorithm", init=False)

    bootstrap_value: bool = field(default=True, init=False)

    make_env_path: Optional[str] = field(default=None, metadata={"help": "Dotted path to custom env factory (e.g. modules.envs.make_env_amp.make_env_amp)"})

    inference_device: str = field(default="cpu", metadata={"help": "InferenceServer device (cpu/cuda/mps)"})
    learner_device: str = field(default="cpu", metadata={"help": "Learner device (cpu/cuda/mps)"})
    num_inference_servers: int = field(default=1, metadata={"help": "Number of InferenceServer processes"})
    # Experiment & environment
    env_id: str = field(default="Humanoid-v5", metadata={"help": "Gym/Gymnasium env id"})
    seed: int = field(default=1, metadata={"help": "Random seed"})
    total_env_steps: int = field(default=2_000_000, metadata={"help": "Total environment steps"})  # note: 1_000_000

    max_policy_lag: int = field(default=0, metadata={"help": "Discard trajectories with policy lag > this (0 = disabled)"})

    # Topology
    num_workers: int = field(default=8, metadata={"help": "Number of rollout worker processes"})
    num_envs_per_worker: int = field(default=2, metadata={"help": "Environments per worker"})
    rollout: int = field(default=64, metadata={"help": "Unroll horizon (steps per trajectory)"})

    # policy
    mlp_layers: List[int] = field(default_factory=lambda: [256, 256, 256], metadata={"help": "MLP layers"})
    use_lstm: bool = field(default=False, metadata={"help": "Use LSTM in policy"})
    lstm_hidden_size: int = field(default=128, metadata={"help": "Hidden size for LSTM"})
    recurrence: int = field(
        default=-1,
        metadata={
            "help": (
                "Trajectory length for backpropagation through time. Default value (-1) "
                "matches the rollout length."
            )
        },
    )

    # RL specifics
    gamma: float = field(default=0.99, metadata={"help": "Discount factor"})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda"})
    replay_capacity: int = field(default=1024, metadata={"help": "Max number of transitions stored (learner buffer size)"})
    learning_starts: int = field(default=1024, metadata={
        "help": "Number of env steps before starting updates (collect-only)"
    })
    batch_size: int = field(default=1024, metadata={"help": "Learner batch size"})
    minibatch_size: int = field(default=256, metadata={"help": "Learner minibatch size"})
    learning_rate: float = field(default=2.5e-5, metadata={"help": "Learning rate"})
    entropy_coef: float = field(default=0.001, metadata={"help": "Entropy regularization coefficient"})
    value_coef: float = field(default=1.0, metadata={"help": "Value loss coefficient"})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm for clipping"})
    train_epochs: int = field(default=1, metadata={"help": "Training epochs per batch"})
    ppo_clip_range: float = field(default=0.2, metadata={"help": "PPO clip range"})
    ppo_clip_value: float = field(default=1.0, metadata={"help": "PPO value clip range"})

    normalize_adv : bool = field(default=True, metadata={"help": "Normalize advantages"})

    # Logging / checkpoints
    logdir: str = field(default="train_logs", metadata={"help": "Root log directory"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Override run name; default: env + timestamp"})
    eval_interval: int = field(default=10_000, metadata={"help": "Eval every N env steps"})
    checkpoint_interval: int = field(default=0, metadata={"help": "Save checkpoint every N env steps (0 = only on exit)"})

    # ---- helpers ----
    def make_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = int(time.time())
        return f"{self.env_id}_{ts}"

    def validate(self) -> None:
        assert self.num_workers > 0
        assert self.rollout > 0
        assert self.total_env_steps > 0
        assert self.num_envs_per_worker > 0

    def __post_init__(self):
        if getattr(self, "use_lstm", False):
            self.data_record_path = "modules.dataset.data_record.ppo_lstm_data_record.PPOLSTMDataRecord"
            self.policy_path = "modules.policy.ppo_lstm_policy.PPOLSTMPolicy"
            self.algorithm_path = "modules.algos.ppo_lstm.PPOLSTMAlgorithm"
        else:
            self.data_record_path = "modules.dataset.data_record.ppo_data_record.PPODataRecord"
            self.policy_path = "modules.policy.ppo_policy.PPOPolicy"
            self.algorithm_path = "modules.algos.ppo.PPOAlgorithm"
        if int(getattr(self, "recurrence", -1)) == -1:
            self.recurrence = int(self.rollout)
        if int(self.recurrence) <= 0:
            raise ValueError("recurrence must be -1 (for rollout length) or a positive integer")
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
