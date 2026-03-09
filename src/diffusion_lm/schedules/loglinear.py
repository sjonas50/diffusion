"""Log-linear noise schedule as used by MDLM (NeurIPS 2024)."""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion_lm.schedules.base import NoiseSchedule

_EPS = 1e-4  # numerical stability: prevents log(0) at t=1


class LogLinearSchedule(NoiseSchedule):
    """Log-linear noise schedule from MDLM (arXiv:2406.07524).

    Defined via the log-SNR: log-SNR(t) = log(alpha(t) / (1 - alpha(t)))
    decreasing linearly in t. This gives:

        alpha(t) = exp(-sigma(t))
        sigma(t) = -log(1 - (1 - eps) * t)

    Equivalently: alpha(t) = 1 - (1 - eps) * t  ... no, that's linear.
    The MDLM formulation uses:
        alpha(t) = (1 - t) / (1 - (1 - eps) * t)  [simplified]

    We use the numerically stable form:
        log_alpha(t) = log1p(-(1 - eps) * t) - log1p(-t * (1 - eps) + t * eps ... )

    Practical implementation (matching MDLM codebase):
        alpha(t) = exp(log1p(-(1 - eps) * t))

    Boundaries:
        - alpha(0) = exp(0) = 1.0
        - alpha(1) = exp(log(eps)) = eps ≈ 0 (not exactly 0 for stability)

    Args:
        eps: Small constant to prevent exact 0 at t=1. Default 1e-4.
    """

    def __init__(self, eps: float = _EPS) -> None:
        self.eps = eps

    def alpha(self, t: Tensor) -> Tensor:
        # log1p(x) = log(1 + x); log1p(-(1-eps)*t) = log(1 - (1-eps)*t)
        log_alpha = torch.log1p(-(1.0 - self.eps) * t)
        return log_alpha.exp().clamp(0.0, 1.0)

    def alpha_derivative(self, t: Tensor) -> Tensor:
        # d/dt [exp(log1p(-(1-eps)*t))]
        # = exp(log1p(-(1-eps)*t)) * d/dt[log1p(-(1-eps)*t)]
        # = alpha(t) * (-(1-eps) / (1 - (1-eps)*t))
        a = self.alpha(t)
        denominator = (1.0 - (1.0 - self.eps) * t).clamp(min=1e-8)
        return a * (-(1.0 - self.eps) / denominator)
