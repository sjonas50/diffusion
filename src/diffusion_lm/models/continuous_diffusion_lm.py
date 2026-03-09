"""Continuous Embedding Diffusion Language Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.diffusion.continuous import ContinuousDiffusionProcess
from diffusion_lm.models.backbone import BidirectionalTransformer
from diffusion_lm.schedules import build_schedule


class ContinuousDiffusionLM(nn.Module):
    """Continuous embedding diffusion language model.

    Unlike masked diffusion, this model:
    1. Works in embedding space (not token space).
    2. Adds Gaussian noise to token embeddings during training.
    3. Predicts clean embeddings from noisy ones (denoising).
    4. Uses a rounding step to convert embeddings to tokens at inference.
    5. Receives timestep t as input (needed because noise level is not
       observable from the embedding itself, unlike mask density).

    Architecture:
        - Input projection: d_model -> d_model (learned)
        - Timestep embedding: nn.Embedding(1000, d_model)
        - Bidirectional transformer backbone
        - Rounding head: d_model -> V (vocabulary size)

    Training loss: MSE (denoising) + CE (rounding).

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
        self.backbone = BidirectionalTransformer(model_config)
        self.diffusion = ContinuousDiffusionProcess(
            schedule=build_schedule(diffusion_config.schedule_type),
            time_epsilon=diffusion_config.time_epsilon,
        )

        d_model = self.backbone.transformer.config.hidden_size
        vocab_size = self.backbone.transformer.config.vocab_size

        # Learned input projection
        self.input_proj = nn.Linear(d_model, d_model)
        # Lightweight timestep embedding (1000 discrete bins)
        self.timestep_emb = nn.Embedding(1000, d_model)
        # Rounding head: maps denoised embeddings to logits over vocabulary
        self.rounding_head = nn.Linear(d_model, vocab_size, bias=False)

        logger.info(
            f"ContinuousDiffusionLM: d_model={d_model}, vocab_size={vocab_size}"
        )

    def _discretize_timestep(self, t: Tensor) -> Tensor:
        """Map continuous t ∈ [0, 1] to discrete bin index ∈ [0, 999]."""
        return (t * 999).long().clamp(0, 999)

    def round_to_tokens(self, embeddings: Tensor, straight_through: bool = False) -> Tensor:
        """Map continuous embeddings to nearest token via cosine similarity.

        Args:
            embeddings: Predicted clean embeddings, shape (B, L, D).
            straight_through: If True, use straight-through estimator for gradients
                (round in forward, identity in backward). Use during training.

        Returns:
            Token logits (similarities), shape (B, L, V).
        """
        # Embedding table from the transformer
        emb_table = self.backbone.transformer.get_input_embeddings().weight  # (V, D)
        emb_table_norm = F.normalize(emb_table, dim=-1)
        embeddings_norm = F.normalize(embeddings, dim=-1)
        # Cosine similarity as logits: (B, L, V)
        logits = torch.matmul(embeddings_norm, emb_table_norm.T)

        if straight_through:
            # Detach the rounding operation so gradients flow through as-if no rounding
            token_ids = logits.argmax(dim=-1)  # (B, L)
            rounded_emb = emb_table[token_ids]  # (B, L, D)
            embeddings = embeddings + (rounded_emb - embeddings).detach()
            logits = torch.matmul(F.normalize(embeddings, dim=-1), emb_table_norm.T)

        return logits

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        t: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            input_ids: Clean token IDs, shape (B, L).
            attention_mask: Optional padding/bidirectional mask.
            labels: Ignored; kept for HF Trainer compatibility.
            t: Optional explicit timesteps. If None, sampled via antithetic sampling.

        Returns:
            Dict with "loss" (scalar) and "logits" (B, L, V).
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Get clean token embeddings from the transformer's embedding table
        emb_table = self.backbone.transformer.get_input_embeddings()
        x0_emb = emb_table(input_ids)  # (B, L, D)

        # Sample or use provided timesteps
        if t is None:
            t = self.diffusion.sample_timesteps(B, device)

        # Apply Gaussian noise
        x_t, _ = self.diffusion.forward_process(x0_emb, t)

        # Learned input projection
        x_t_proj = self.input_proj(x_t)

        # Add timestep embedding
        t_idx = self._discretize_timestep(t)  # (B,)
        t_emb = self.timestep_emb(t_idx).unsqueeze(1).expand(B, L, -1)  # (B, L, D)
        x_t_proj = x_t_proj + t_emb

        # Bidirectional denoising via transformer
        # We replace input embeddings with our noisy projected embeddings
        # by using inputs_embeds instead of input_ids
        transformer_outputs = self.backbone.transformer(
            inputs_embeds=x_t_proj,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state  # (B, L, D)

        # Predict clean embeddings
        predicted_x0 = hidden_states  # (B, L, D)

        # MSE denoising loss
        mse_loss = self.diffusion.compute_loss(predicted_x0, x0_emb, t)

        # Rounding loss: CE between rounded predictions and true tokens
        rounding_logits = self.round_to_tokens(predicted_x0, straight_through=True)
        rounding_loss = F.cross_entropy(
            rounding_logits.reshape(B * L, -1),
            input_ids.reshape(B * L),
        )

        loss = mse_loss + rounding_loss
        return {"loss": loss, "logits": rounding_logits}
