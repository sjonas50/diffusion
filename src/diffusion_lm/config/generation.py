"""Generation configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class GenerationConfig:
    """Configuration for diffusion LM generation / sampling.

    Args:
        num_steps: Number of denoising steps. More steps = better quality, slower.
        temperature: Sampling temperature applied to logits before argmax.
        sampler: Which sampler to use. "first_hitting" is the default —
            theoretically correct (arXiv:2409.02908) and 20x faster than
            confidence-based categorical sampling.
        block_size: Block size for BD3LM block sampler. None = auto from model config.
        remasking: Token selection strategy at each step (only applies when
            sampler="first_hitting" uses confidence threshold).
            - "confidence": Unmask most-confident positions first.
            - "random": Unmask random positions.
        running_confidence_remasking: Enable Running Confidence Remasking to prevent
            "Answer Backslide" — correct intermediate tokens getting overwritten.
            Free, no retraining required. Enabled by default.
        max_new_tokens: Maximum tokens to generate beyond the prompt.
        guidance_scale: Classifier-free guidance scale. 0.0 = no guidance.
    """

    num_steps: int = 64
    temperature: float = 1.0
    sampler: Literal["first_hitting", "block", "continuous", "cached"] = "first_hitting"
    block_size: int | None = None
    remasking: Literal["confidence", "random"] = "confidence"
    running_confidence_remasking: bool = True
    max_new_tokens: int = 128
    guidance_scale: float = 0.0
