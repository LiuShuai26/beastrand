import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.kl import kl_divergence


class DiagGaussianDistribution(nn.Module):
    def __init__(self, in_dim: int, act_dim: int,
                 init_log_std: float = 0.0,
                 min_log_std: float = -5.0,
                 max_log_std: float = 2.0):
        super().__init__()
        self.mean = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std, requires_grad=True)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self._dist = None

    def forward(self, feats: torch.Tensor):
        mean = self.mean(feats)  # [B, A]
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp().expand_as(mean)  # [B, A]
        self._dist = Normal(mean, std)
        return self

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.mode() if deterministic else self.sample()

    def sample(self) -> torch.Tensor:
        return self._dist.rsample()

    def mode(self) -> torch.Tensor:
        return self._dist.loc

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions).sum(-1)  # [B]

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(-1)  # [B]

    def action_logits(self) -> torch.Tensor:
        """Return concat(mean, log_std_expanded) as [B, act_dim*2].

        Used to store the old policy's distribution parameters in the
        trajectory buffer so that the learner can compute the analytical
        KL divergence KL(π_new ‖ π_old) during PPO updates.
        """
        mean = self._dist.loc  # [B, A]
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        return torch.cat([mean, log_std.expand_as(mean)], dim=-1)

    @staticmethod
    def kl_from_logits(logits_new: torch.Tensor,
                       logits_old: torch.Tensor) -> torch.Tensor:
        """Analytical KL(π_new ‖ π_old) for diagonal Gaussian, per sample.

        Args:
            logits_new: [B, act_dim*2] concat(mean, log_std) from current policy.
            logits_old: [B, act_dim*2] concat(mean, log_std) stored from old policy.

        Returns:
            [B] per-sample KL divergence (summed over action dimensions).
        """
        d = logits_new.shape[-1] // 2
        mu_new, logstd_new = logits_new[..., :d], logits_new[..., d:]
        mu_old, logstd_old = logits_old[..., :d], logits_old[..., d:]
        dist_new = Normal(mu_new, logstd_new.exp())
        dist_old = Normal(mu_old, logstd_old.exp())
        return kl_divergence(dist_new, dist_old).sum(-1)  # [B]


class CategoricalDistribution(nn.Module):
    def __init__(self, in_dim: int, num_actions: int):
        super().__init__()
        self.logits = nn.Linear(in_dim, num_actions)
        self._dist = None

    def forward(self, feats: torch.Tensor):
        self._dist = Categorical(logits=self.logits(feats))
        return self

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.mode() if deterministic else self.sample()

    def sample(self) -> torch.Tensor:
        return self._dist.sample()  # [B] long

    def mode(self) -> torch.Tensor:
        return self._dist.probs.argmax(-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)
        return self._dist.log_prob(actions)  # [B]

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()  # [B]
