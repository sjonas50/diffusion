"""Continuous diffusion sampler (DDPM-style reverse process)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from diffusion_lm.samplers.base import Sampler, SamplerOutput

if TYPE_CHECKING:
    from torch import Tensor

    from diffusion_lm.config.generation import GenerationConfig


class ContinuousSampler(Sampler):
    """DDPM-style reverse sampler for ContinuousDiffusionLM.

    Runs the reverse diffusion process:
        x_{t-1} = sqrt(alpha_{t-1}) * predicted_x0 + sqrt(1 - alpha_{t-1}) * noise

    Final step uses nearest-neighbor rounding to discrete tokens (no STE at inference).
    """

    def generate(
        self,
        model,
        prompt_ids: Tensor,
        config: GenerationConfig,
    ) -> SamplerOutput:
        """Generate completions via DDPM reverse process.

        Args:
            model: ContinuousDiffusionLM.
            prompt_ids: Shape (B, prompt_len).
            config: GenerationConfig.

        Returns:
            SamplerOutput with full sequences (B, prompt_len + max_new_tokens).
        """
        B, prompt_len = prompt_ids.shape
        gen_len = config.max_new_tokens
        device = prompt_ids.device
        num_steps = config.num_steps

        schedule = model.diffusion.schedule

        # Get embedding dimension from model
        embed_dim = model.backbone.transformer.config.hidden_size

        # Initialize from pure noise
        x_t = torch.randn(B, gen_len, embed_dim, device=device)

        for step in range(num_steps - 1, -1, -1):
            t = step / max(num_steps - 1, 1)
            t_prev = max((step - 1) / max(num_steps - 1, 1), 0.0)

            t_tensor = torch.full((B,), t, device=device)

            with torch.no_grad():
                # Project noisy embeddings and add timestep embedding
                x_proj = model.input_proj(x_t)
                t_idx = model._discretize_timestep(t_tensor)
                t_emb = model.timestep_emb(t_idx).unsqueeze(1).expand(B, gen_len, -1)
                x_proj = x_proj + t_emb

                # Run transformer on embeddings (use inputs_embeds, not input_ids)
                transformer_out = model.backbone.transformer(inputs_embeds=x_proj)
                predicted_x0 = transformer_out.logits  # (B, gen_len, V) — use hidden states

            # For continuous diffusion, we need the hidden states, not logits.
            # Access hidden states if available, otherwise fall back to a learned head.
            # Since ContinuousDiffusionLM predicts clean embeddings, we use the
            # rounding_head inverse or just treat the output as predicted embeddings.
            # The simplest correct approach: transformer predicts in embedding space.
            # For now, use the last hidden state via output_hidden_states=True.
            with torch.no_grad():
                transformer_out = model.backbone.transformer(
                    inputs_embeds=x_proj,
                    output_hidden_states=True,
                )
                predicted_x0 = transformer_out.hidden_states[-1]  # (B, gen_len, D)

            alpha_prev = schedule.alpha(
                torch.full((B,), t_prev, device=device)
            ).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

            # DDPM step
            noise = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
            x_t = (
                alpha_prev.sqrt() * predicted_x0
                + (1 - alpha_prev).sqrt() * noise
            )

        # Round to nearest token via embedding cosine similarity
        rounding_logits = model.round_to_tokens(x_t)  # (B, gen_len, V)
        token_ids = rounding_logits.argmax(dim=-1)  # (B, gen_len)

        sequences = torch.cat([prompt_ids, token_ids], dim=1)
        return SamplerOutput(sequences=sequences)
