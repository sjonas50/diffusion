"""Pretraining trainer with random truncation."""

from __future__ import annotations

from diffusion_lm.data.collators import RandomTruncateCollator
from diffusion_lm.trainers.base import DiffusionTrainer


class PretrainingTrainer(DiffusionTrainer):
    """Diffusion trainer for pretraining with random truncation.

    Automatically uses RandomTruncateCollator if no data_collator is provided.
    Random truncation (default 1% of batches) improves robustness to variable-
    length sequences at inference (LLaDA technique).
    """

    def __init__(self, pad_token_id: int = 0, **kwargs) -> None:
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            training_args = kwargs.get("args")
            ratio = getattr(training_args, "random_truncation_ratio", 0.01)
            kwargs["data_collator"] = RandomTruncateCollator(
                pad_token_id=pad_token_id,
                truncation_ratio=ratio,
            )
        super().__init__(**kwargs)
