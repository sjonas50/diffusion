"""Abstract base sampler for diffusion LM generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from diffusion_lm.config.generation import GenerationConfig


@dataclass
class SamplerOutput:
    """Output of a sampler's generate() call.

    Args:
        sequences: Full token sequences (prompt + completion), shape (B, L).
        scores: Optional per-token log-probabilities, shape (B, gen_len).
    """

    sequences: Tensor
    scores: Tensor | None = None


class Sampler(ABC):
    """Abstract base class for diffusion LM samplers."""

    @abstractmethod
    def generate(
        self,
        model,
        prompt_ids: Tensor,
        config: GenerationConfig,
    ) -> SamplerOutput:
        """Generate completions for the given prompts.

        Args:
            model: MaskedDiffusionLM (or compatible) model.
            prompt_ids: Prompt token IDs, shape (B, prompt_len).
            config: Generation configuration.

        Returns:
            SamplerOutput with sequences of shape (B, prompt_len + max_new_tokens).
        """
