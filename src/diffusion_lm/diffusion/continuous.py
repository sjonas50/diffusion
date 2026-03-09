"""Continuous embedding diffusion process (Diffusion-LM style)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_lm.diffusion.base import DiffusionProcess
from diffusion_lm.schedules.base import NoiseSchedule


class ContinuousDiffusionProcess(DiffusionProcess):
    """Gaussian noise diffusion over token embedding space.

    Forward process adds Gaussian noise to token embeddings using the DDPM schedule:
        x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * epsilon
    where epsilon ~ N(0, I).

    Training objective: predict clean embeddings x_0 from noisy x_t (MSE loss).
    Inference: iterative DDPM denoising followed by rounding to nearest token.

    Note: This is the only diffusion variant where the model receives timestep t
    as input (unlike masked diffusion where t is inferable from mask density).

    Args:
        schedule: Noise schedule. Here alpha(t) plays the role of alpha_bar (SNR proxy).
        time_epsilon: Minimum timestep.
    """

    def __init__(self, schedule: NoiseSchedule, time_epsilon: float = 1e-3) -> None:
        super().__init__(schedule)
        self.time_epsilon = time_epsilon

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Antithetic timestep sampling scaled to [time_epsilon, 1)."""
        t = super().sample_timesteps(batch_size, device)
        return t * (1.0 - self.time_epsilon) + self.time_epsilon

    def forward_process(
        self, x0_emb: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply Gaussian forward process.

        Args:
            x0_emb: Clean token embeddings, shape (B, L, D).
            t: Timesteps, shape (B,).

        Returns:
            x_t: Noisy embeddings, shape (B, L, D).
            noise: Gaussian noise added, shape (B, L, D).
        """
        # alpha_bar(t) = alpha(t) here (schedule already provides the right value)
        alpha_bar = self.schedule.alpha(t).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        noise = torch.randn_like(x0_emb)
        x_t = torch.sqrt(alpha_bar) * x0_emb + torch.sqrt(1.0 - alpha_bar + 1e-8) * noise
        return x_t, noise

    def compute_loss(
        self,
        predicted_x0: Tensor,
        x0_emb: Tensor,
        t: Tensor,
    ) -> Tensor:
        """MSE loss between predicted and actual clean embeddings.

        Args:
            predicted_x0: Denoised embedding predictions, shape (B, L, D).
            x0_emb: True clean embeddings, shape (B, L, D).
            t: Timesteps (unused in MSE but kept for API consistency).

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(predicted_x0, x0_emb)
