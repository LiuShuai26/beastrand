# projects/ppo_lstm/config.py
"""PPO-LSTM configuration. Inherits PPO config, adds LSTM-specific fields."""

from dataclasses import dataclass, field

from ppo.config import Args as PPOArgs


@dataclass
class Args(PPOArgs):

    data_record_path: str = field(default="projects.ppo_lstm.data_record.PPOLSTMDataRecord", init=False)
    policy_path: str = field(default="projects.ppo_lstm.policy.PPOLSTMPolicy", init=False)
    algorithm_path: str = field(default="projects.ppo_lstm.algorithm.PPOLSTMAlgorithm", init=False)

    # LSTM-specific
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

    def __post_init__(self):
        # Resolve recurrence before parent validation
        if int(getattr(self, "recurrence", -1)) == -1:
            self.recurrence = int(self.rollout)
        if int(self.recurrence) <= 0:
            raise ValueError("recurrence must be -1 (for rollout length) or a positive integer")
        super().__post_init__()
