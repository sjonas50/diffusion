"""Diffusion process configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion process.

    Args:
        process_type: Diffusion variant to use.
            - "masked": Standard masked diffusion (LLaDA/Mercury style).
            - "block": Block diffusion / BD3LM (AR over blocks, masked within).
            - "continuous": Gaussian noise over token embeddings.
        schedule_type: Noise schedule controlling masking probability alpha(t).
        mask_token_id: Token ID for the [MASK] token. Must be set before training
            (default -1 is invalid — raises ValueError in trainer init).
        time_epsilon: Minimum timestep to avoid t=0 (fully clean) during training.
        block_size: Block size for BD3LM block diffusion. Only used when
            process_type="block". None means process all tokens as one block.
    """

    process_type: Literal["masked", "block", "continuous"] = "masked"
    schedule_type: Literal["linear", "cosine", "loglinear"] = "linear"
    mask_token_id: int = -1
    time_epsilon: float = 1e-3
    block_size: int | None = None
