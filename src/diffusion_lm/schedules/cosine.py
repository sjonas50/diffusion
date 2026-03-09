"""Cosine noise schedule: alpha(t) = cos(pi/2 * t)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from diffusion_lm.schedules.base import NoiseSchedule


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule.

    alpha(t) = cos(pi/2 * t)
    - alpha(0) = 1.0 (clean)
    - alpha(1) = 0.0 (fully masked)
    - alpha'(t) = -pi/2 * sin(pi/2 * t) ≤ 0
    """

    def alpha(self, t: Tensor) -> Tensor:
        return torch.cos(math.pi / 2.0 * t).clamp(0.0, 1.0)

    def alpha_derivative(self, t: Tensor) -> Tensor:
        return -(math.pi / 2.0) * torch.sin(math.pi / 2.0 * t)
