"""SFT trainer with prompt-aware masking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from diffusion_lm.data.collators import SFTCollator
from diffusion_lm.trainers.base import DiffusionTrainer

if TYPE_CHECKING:
    from torch import Tensor


class SFTTrainer(DiffusionTrainer):
    """Diffusion trainer for supervised fine-tuning.

    Ensures prompt_mask is present in every batch, protecting prompt tokens
    from being masked and excluding them from the training loss.

    Automatically uses SFTCollator if no data_collator is provided.
    """

    def __init__(self, pad_token_id: int = 0, **kwargs) -> None:
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            kwargs["data_collator"] = SFTCollator(pad_token_id=pad_token_id)
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Validate prompt_mask presence and delegate to parent."""
        if "prompt_mask" not in inputs:
            raise ValueError(
                "SFTTrainer requires 'prompt_mask' in batch inputs. "
                "Ensure your dataset produces prompt_mask and SFTCollator is used."
            )
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
