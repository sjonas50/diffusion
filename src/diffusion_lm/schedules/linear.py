"""Linear noise schedule: alpha(t) = 1 - t."""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion_lm.schedules.base import NoiseSchedule


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule used by LLaDA.

    alpha(t) = 1 - t
    - alpha(0) = 1.0 (clean)
    - alpha(1) = 0.0 (fully masked)
    - alpha'(t) = -1 (constant)
    - weight(t) = 1 / t
    """

    def alpha(self, t: Tensor) -> Tensor:
        return (1.0 - t).clamp(0.0, 1.0)

    def alpha_derivative(self, t: Tensor) -> Tensor:
        return torch.full_like(t, -1.0)
