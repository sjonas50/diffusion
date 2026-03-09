"""Abstract base class for noise schedules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class NoiseSchedule(ABC):
    """Abstract noise schedule mapping timestep t ∈ [0, 1] to masking probabilities.

    Convention:
        - alpha(t) = "keep probability" = probability a token is NOT masked.
        - alpha(0) = 1.0 (fully clean: no tokens masked).
        - alpha(1) = 0.0 (fully corrupted: all tokens masked).
        - alpha is monotonically decreasing.
    """

    @abstractmethod
    def alpha(self, t: Tensor) -> Tensor:
        """Compute keep probability at timestep t.

        Args:
            t: Timesteps in [0, 1], arbitrary shape.

        Returns:
            Keep probabilities with same shape as t, in [0, 1].
        """
        ...

    @abstractmethod
    def alpha_derivative(self, t: Tensor) -> Tensor:
        """Compute d/dt alpha(t).

        Args:
            t: Timesteps in [0, 1], arbitrary shape.

        Returns:
            Derivative values with same shape as t. Should be ≤ 0 everywhere.
        """
        ...

    def mask_probability(self, t: Tensor) -> Tensor:
        """Masking probability at timestep t: p_mask(t) = 1 - alpha(t).

        Args:
            t: Timesteps in [0, 1], arbitrary shape.

        Returns:
            Masking probabilities with same shape as t, in [0, 1].
        """
        return 1.0 - self.alpha(t)

    def weight(self, t: Tensor) -> Tensor:
        """ELBO loss weight at timestep t: w(t) = -alpha'(t) / (1 - alpha(t)).

        This weight makes the expected loss an unbiased estimate of the ELBO.
        Always positive since alpha'(t) ≤ 0 and (1 - alpha(t)) ≥ 0.

        Args:
            t: Timesteps in (0, 1), arbitrary shape.

        Returns:
            Positive weights with same shape as t.
        """
        denom = (1.0 - self.alpha(t)).clamp(min=1e-5)
        return (-self.alpha_derivative(t)) / denom
