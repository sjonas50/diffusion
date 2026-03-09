"""Masked Diffusion Language Model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.diffusion.block import BlockDiffusionProcess
from diffusion_lm.diffusion.masked import MaskedDiffusionProcess
from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token
from diffusion_lm.schedules import build_schedule


class MaskedDiffusionLM(nn.Module):
    """Full masked diffusion language model.

    Composes BidirectionalTransformer + MaskedDiffusionProcess (or BlockDiffusionProcess).

    During training:
        1. Sample timesteps t via antithetic sampling.
        2. Apply forward process: randomly mask tokens.
        3. Protect prompt tokens (SFT mode): restore prompt positions.
        4. Run bidirectional transformer on corrupted sequence.
        5. Compute ELBO-weighted CE loss over masked (response) positions.

    Args:
        model_config: Backbone configuration.
        diffusion_config: Diffusion process configuration.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        super().__init__()

        if diffusion_config.mask_token_id == -1:
            raise ValueError(
                "DiffusionConfig.mask_token_id is -1 (unset). "
                "Call add_mask_token(model, tokenizer) and set mask_token_id before training."
            )

        self.backbone = BidirectionalTransformer(model_config)
        schedule = build_schedule(diffusion_config.schedule_type)

        if diffusion_config.process_type == "block":
            block_size = diffusion_config.block_size or 64
            self.diffusion = BlockDiffusionProcess(
                schedule=schedule,
                mask_token_id=diffusion_config.mask_token_id,
                block_size=block_size,
                time_epsilon=diffusion_config.time_epsilon,
            )
        else:
            self.diffusion = MaskedDiffusionProcess(
                schedule=schedule,
                mask_token_id=diffusion_config.mask_token_id,
                time_epsilon=diffusion_config.time_epsilon,
            )

    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
        tokenizer=None,
    ) -> MaskedDiffusionLM:
        """Create model and optionally add mask token.

        If tokenizer is provided and mask_token_id is -1, adds [MASK] token
        and updates diffusion_config.mask_token_id automatically.
        """
        if tokenizer is not None and diffusion_config.mask_token_id == -1:
            # Need a temporary backbone to resize embeddings
            backbone = BidirectionalTransformer(model_config)
            mask_token_id = add_mask_token(backbone, tokenizer)
            diffusion_config.mask_token_id = mask_token_id
            # Now create the full model (backbone will be re-initialized)
            # TODO: avoid double initialization in a future refactor
        return cls(model_config, diffusion_config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        prompt_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            input_ids: Clean token IDs, shape (B, L).
            attention_mask: Padding mask or 4D bidirectional mask, shape (B, L) or (B, 1, L, L).
            labels: Ignored (targets are derived from input_ids). Kept for HF Trainer compat.
            prompt_mask: Optional boolean mask, shape (B, L).
                True = prompt position (never masked, excluded from loss).
                None = standard pretraining (all positions participate).

        Returns:
            Dict with "loss" (scalar) and "logits" (B, L, V).
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Sample timesteps (antithetic stratified sampling)
        t = self.diffusion.sample_timesteps(B, device)

        # Forward corruption
        corrupted, _ = self.diffusion.forward_process(input_ids, t)

        # SFT mode: restore prompt tokens (never masked)
        if prompt_mask is not None:
            corrupted = torch.where(prompt_mask, input_ids, corrupted)

        # Run bidirectional transformer
        logits = self.backbone(corrupted, attention_mask)

        # Compute ELBO-weighted loss (only over masked positions)
        loss_mask = ~prompt_mask if prompt_mask is not None else None
        loss = self.diffusion.compute_loss(logits, input_ids, corrupted, t, loss_mask)

        return {"loss": loss, "logits": logits}
