"""
AMP (Adversarial Motion Priors) training components.

- discriminator: Style discriminator network with gradient penalty
- motion_buffer: Motion clip loading and random frame sampling
- rewards: Style reward computation from discriminator output
"""

from .discriminator import AMPDiscriminator
from .motion_buffer import AMPMotionBuffer
from .rewards import compute_disc_loss, compute_style_reward
