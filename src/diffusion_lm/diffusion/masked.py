"""Masked diffusion process (LLaDA/Mercury style)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_lm.diffusion.base import DiffusionProcess
from diffusion_lm.schedules.base import NoiseSchedule


class MaskedDiffusionProcess(DiffusionProcess):
    """Masked diffusion process for discrete token sequences.

    Forward process: randomly mask tokens with probability p_mask(t) = 1 - alpha(t).
    Training objective: ELBO-weighted cross-entropy on masked positions.

    Loss = CE(logits[masked], targets[masked]) / p_mask(t)
    This weighting makes the loss an unbiased estimate of the ELBO upper bound
    on the negative log-likelihood (proven in LLaDA paper).

    Args:
        schedule: Noise schedule controlling masking probability.
        mask_token_id: Token ID for the [MASK] token.
        time_epsilon: Minimum timestep (avoids t=0 / no masking edge case).
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        mask_token_id: int,
        time_epsilon: float = 1e-3,
    ) -> None:
        super().__init__(schedule)
        self.mask_token_id = mask_token_id
        self.time_epsilon = time_epsilon

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Antithetic timestep sampling scaled to [time_epsilon, 1)."""
        t = super().sample_timesteps(batch_size, device)
        # Scale from [0, 1) to [time_epsilon, 1)
        return t * (1.0 - self.time_epsilon) + self.time_epsilon

    def forward_process(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Apply forward masking process.

        Args:
            x0: Clean token IDs, shape (B, L).
            t: Timesteps, shape (B,). Each t_i ∈ (time_epsilon, 1).

        Returns:
            corrupted: Token IDs with masked positions replaced by mask_token_id,
                shape (B, L).
            token_mask: Boolean mask of masked positions, shape (B, L).
                True where token is masked.
        """
        B, L = x0.shape
        # p_mask(t) = 1 - alpha(t), broadcast to (B, L)
        p_mask = self.schedule.mask_probability(t).unsqueeze(-1).expand(B, L)
        # Independent Bernoulli mask per token
        token_mask = torch.rand_like(p_mask) < p_mask
        corrupted = torch.where(token_mask, torch.full_like(x0, self.mask_token_id), x0)
        return corrupted, token_mask

    def compute_loss(
        self,
        logits: Tensor,
        x0: Tensor,
        corrupted: Tensor,
        t: Tensor,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute ELBO-weighted cross-entropy loss.

        Only computes loss over masked positions. The weighting by 1/p_mask(t)
        makes this an unbiased ELBO estimate.

        For SFT: pass loss_mask=~prompt_mask to restrict loss to response tokens.

        Args:
            logits: Model output logits, shape (B, L, V).
            x0: Original clean token IDs, shape (B, L).
            corrupted: Corrupted token IDs (output of forward_process), shape (B, L).
            t: Timesteps, shape (B,).
            loss_mask: Optional boolean mask of positions to include in loss,
                shape (B, L). True = include. None = all positions.

        Returns:
            Scalar loss value.
        """
        B, L, V = logits.shape

        # Positions that are masked (and therefore have a training signal)
        masked_positions = corrupted == self.mask_token_id  # (B, L)

        # Optionally restrict to a subset (e.g. response tokens in SFT)
        if loss_mask is not None:
            masked_positions = masked_positions & loss_mask

        n_masked = masked_positions.sum().clamp(min=1)
        if n_masked == 0:
            return logits.sum() * 0.0  # differentiable zero

        # Per-token cross-entropy, shape (B, L)
        ce = F.cross_entropy(
            logits.reshape(B * L, V),
            x0.reshape(B * L),
            reduction="none",
        ).reshape(B, L)

        # ELBO weighting: divide by p_mask(t) per sample, broadcast to (B, L)
        # Without this, low-noise timesteps (few masks) are undertrained
        p_mask = self.schedule.mask_probability(t).unsqueeze(-1).clamp(min=1e-5)  # (B, 1)
        weighted_ce = ce / p_mask  # (B, L)

        # Sum over masked positions, normalize by count
        loss = (weighted_ce * masked_positions.float()).sum() / n_masked
        return loss
