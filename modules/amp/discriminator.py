"""
Style discriminator for Adversarial Motion Priors (AMP).

A neural network that learns to classify motion states as "real" (from
reference motion data) or "fake" (from the RL policy). Trained with
gradient penalty for Lipschitz constraint stability.

Reference: Peng et al., "AMP: Adversarial Motion Priors for Stylized
Physics-Based Character Animation", SIGGRAPH 2021.
"""

import numpy as np
import torch
import torch.nn as nn


def _disc_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AMPDiscriminator(nn.Module):
    """MLP discriminator for AMP style reward.

    Input:  AMP observation (joint angles + relative body positions).
    Output: Single logit — positive for real, negative for fake.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(_disc_layer_init(nn.Linear(prev_dim, hidden_dim)))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(_disc_layer_init(nn.Linear(prev_dim, 1), std=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, amp_obs):
        """Forward pass. Returns logits of shape (B, 1)."""
        return self.net(amp_obs)

    def compute_grad_penalty(self, amp_obs):
        """Compute gradient penalty (R1 regularisation).

        Penalises ||nabla_x D(x)||^2, encouraging Lipschitz smoothness
        around the real data manifold.

        Args:
            amp_obs: (B, obs_dim) tensor — gradient will be enabled internally.

        Returns:
            Scalar gradient penalty loss.
        """
        amp_obs = amp_obs.detach().requires_grad_(True)
        logits = self.forward(amp_obs)
        grad = torch.autograd.grad(
            logits.sum(), amp_obs, create_graph=True
        )[0]
        return (grad.norm(2, dim=-1) ** 2).mean()
