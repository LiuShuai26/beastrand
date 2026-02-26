"""
Style reward computation from AMP discriminator output.

Standard AMP formulation:
    r_style = -log(1 - sigmoid(D(s))) = softplus(D(s))
"""

import torch
import torch.nn.functional as F


def compute_style_reward(disc_logits):
    """Compute AMP style reward from discriminator logits.

    Uses softplus (= -log(1 - sigmoid(x))) clamped for stability.

    Args:
        disc_logits: Discriminator output logits, shape (B,) or (B, 1).

    Returns:
        Style rewards, shape (B,).
    """
    if disc_logits.dim() > 1:
        disc_logits = disc_logits.squeeze(-1)
    return torch.clamp(F.softplus(disc_logits), max=10.0)


def compute_disc_loss(disc_real_logits, disc_fake_logits):
    """Compute discriminator binary cross-entropy loss.

    Real (motion data)  -> label 1.
    Fake (policy data)  -> label 0.

    Args:
        disc_real_logits: (B, 1) logits for reference motion samples.
        disc_fake_logits: (B, 1) logits for policy-generated samples.

    Returns:
        Scalar loss.
    """
    real_loss = F.binary_cross_entropy_with_logits(
        disc_real_logits, torch.ones_like(disc_real_logits)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        disc_fake_logits, torch.zeros_like(disc_fake_logits)
    )
    return real_loss + fake_loss
