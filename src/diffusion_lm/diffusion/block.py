"""Block diffusion process — BD3LM (arXiv:2503.09573, ICLR 2025 Oral).

Autoregressive over blocks, masked diffusion within each block.
Enables KV caching for completed blocks during inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_lm.diffusion.base import DiffusionProcess
from diffusion_lm.schedules.base import NoiseSchedule


class BlockDiffusionProcess(DiffusionProcess):
    """BD3LM block diffusion process.

    The sequence is divided into non-overlapping blocks of size `block_size`.
    - Autoregressive across blocks: earlier blocks are kept clean (fully visible).
    - Masked diffusion within the current block: tokens corrupted independently.

    During training, a random block index is chosen per sample. All blocks
    before it are clean context; the selected block is masked with probability
    p_mask(t); blocks after it are fully masked.

    Args:
        schedule: Noise schedule for within-block masking.
        mask_token_id: Token ID for [MASK].
        block_size: Number of tokens per block.
        time_epsilon: Minimum timestep.
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        mask_token_id: int,
        block_size: int = 64,
        time_epsilon: float = 1e-3,
    ) -> None:
        super().__init__(schedule)
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.time_epsilon = time_epsilon

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Antithetic timestep sampling scaled to [time_epsilon, 1)."""
        t = super().sample_timesteps(batch_size, device)
        return t * (1.0 - self.time_epsilon) + self.time_epsilon

    def _get_block_boundaries(self, seq_len: int) -> list[tuple[int, int]]:
        """Return (start, end) index pairs for each block."""
        boundaries = []
        start = 0
        while start < seq_len:
            end = min(start + self.block_size, seq_len)
            boundaries.append((start, end))
            start = end
        return boundaries

    def forward_process(
        self, x0: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply block-structured forward masking.

        For each sample in the batch:
        1. Divide the sequence into blocks.
        2. Randomly select one block as the "active" block.
        3. Blocks before the active block: kept clean (AR context).
        4. Active block: masked with probability p_mask(t).
        5. Blocks after the active block: fully masked.

        Args:
            x0: Clean token IDs, shape (B, L).
            t: Timesteps, shape (B,).

        Returns:
            corrupted: Masked token IDs, shape (B, L).
            token_mask: Boolean mask of masked positions, shape (B, L).
        """
        B, L = x0.shape
        boundaries = self._get_block_boundaries(L)
        n_blocks = len(boundaries)

        corrupted = x0.clone()
        token_mask = torch.zeros(B, L, dtype=torch.bool, device=x0.device)

        # Per-sample masking probability
        p_mask = self.schedule.mask_probability(t)  # (B,)

        # Randomly select active block per sample
        active_block = torch.randint(0, n_blocks, (B,), device=x0.device)

        for block_idx, (start, end) in enumerate(boundaries):
            block_len = end - start

            for b in range(B):
                if block_idx < active_block[b]:
                    # Before active block: clean (AR context)
                    pass
                elif block_idx == active_block[b]:
                    # Active block: mask with probability p_mask(t)
                    mask_probs = torch.full(
                        (block_len,), p_mask[b].item(), device=x0.device
                    )
                    block_mask = torch.rand(block_len, device=x0.device) < mask_probs
                    token_mask[b, start:end] = block_mask
                    corrupted[b, start:end] = torch.where(
                        block_mask,
                        torch.full_like(x0[b, start:end], self.mask_token_id),
                        x0[b, start:end],
                    )
                else:
                    # After active block: fully masked
                    token_mask[b, start:end] = True
                    corrupted[b, start:end] = self.mask_token_id

        return corrupted, token_mask

    def compute_loss(
        self,
        logits: Tensor,
        x0: Tensor,
        corrupted: Tensor,
        t: Tensor,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        """ELBO-weighted CE loss over masked positions.

        Same as MaskedDiffusionProcess but applied to block-structured masking.

        Args:
            logits: Model output logits, shape (B, L, V).
            x0: Clean token IDs, shape (B, L).
            corrupted: Masked token IDs, shape (B, L).
            t: Timesteps, shape (B,).
            loss_mask: Optional restriction mask, shape (B, L).

        Returns:
            Scalar loss.
        """
        B, L, V = logits.shape
        masked_positions = corrupted == self.mask_token_id
        if loss_mask is not None:
            masked_positions = masked_positions & loss_mask

        n_masked = masked_positions.sum().clamp(min=1)

        ce = F.cross_entropy(
            logits.reshape(B * L, V),
            x0.reshape(B * L),
            reduction="none",
        ).reshape(B, L)

        p_mask = self.schedule.mask_probability(t).unsqueeze(-1).clamp(min=1e-5)
        weighted_ce = ce / p_mask
        return (weighted_ce * masked_positions.float()).sum() / n_masked
