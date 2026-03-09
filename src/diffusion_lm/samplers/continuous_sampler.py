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

        # Initialize from pure noise
        embed_dim = model.input_proj.out_features
        x_t = torch.randn(B, gen_len, embed_dim, device=device)

        for step in range(num_steps - 1, -1, -1):
            t = step / max(num_steps - 1, 1)
            t_prev = max((step - 1) / max(num_steps - 1, 1), 0.0)

            t_tensor = torch.full((B,), t, device=device)

            with torch.no_grad():
                # Predict x0 embedding from noisy x_t
                predicted_x0 = model.backbone(x_t, t_tensor)  # (B, gen_len, embed_dim)

            alpha_prev = schedule.alpha(
                torch.full((B,), t_prev, device=device)
            ).unsqueeze(-1)

            # DDPM step
            noise = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
            x_t = (
                alpha_prev.sqrt() * predicted_x0
                + (1 - alpha_prev).sqrt() * noise
            )

        # Round to nearest token via embedding lookup
        token_ids = model.round_to_tokens(x_t)  # (B, gen_len)

        sequences = torch.cat([prompt_ids, token_ids], dim=1)
        return SamplerOutput(sequences=sequences)
