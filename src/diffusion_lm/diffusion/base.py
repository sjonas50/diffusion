"""Abstract base class for diffusion processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from diffusion_lm.schedules.base import NoiseSchedule


class DiffusionProcess(ABC):
    """Abstract diffusion process defining forward corruption and loss computation."""

    def __init__(self, schedule: NoiseSchedule) -> None:
        self.schedule = schedule

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timesteps using antithetic (stratified) sampling.

        Instead of B independent uniform samples, draws one u ~ U[0, 1) and
        creates t_i = (u + i/B) mod 1 scaled to [time_epsilon, 1). This ensures
        every batch covers the full noise range, reducing gradient variance by ~B-fold.

        Args:
            batch_size: Number of timesteps to sample (one per sequence in batch).
            device: Target device.

        Returns:
            Timesteps of shape (batch_size,) in (time_epsilon, 1).
        """
        u = torch.rand(1, device=device)
        indices = torch.arange(batch_size, device=device, dtype=torch.float32)
        t = ((u + indices / batch_size) % 1.0).float()
        return t

    @abstractmethod
    def forward_process(self, *args, **kwargs):
        """Apply forward corruption to clean data."""
        ...

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Tensor:
        """Compute training loss from model outputs."""
        ...
