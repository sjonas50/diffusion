"""Base diffusion trainer extending HuggingFace Trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import TrainingArguments


class NanLossCallback(TrainerCallback):
    """Stop training and log diagnostics when NaN loss is detected.

    LLaDA experienced NaN crashes at 1.2T/2.3T tokens. This callback
    provides early detection with a clear error message.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs and "loss" in logs:
            loss_val = logs["loss"]
            if loss_val != loss_val or loss_val == float("inf"):  # NaN or Inf check
                logger.error(
                    f"NaN/Inf loss detected at step {state.global_step}! "
                    "Stopping training. Consider: reducing LR, increasing grad clip, "
                    "or checking for corrupted data batches."
                )
                control.should_training_stop = True


class DiffusionTrainer(Trainer):
    """HuggingFace Trainer extended for diffusion LM training.

    Overrides only compute_loss() to call the diffusion model's forward pass.
    All other functionality (gradient accumulation, mixed precision, DDP/FSDP/
    DeepSpeed, checkpointing, W&B logging) is inherited from HF Trainer.

    Usage:
        trainer = DiffusionTrainer(
            model=masked_diffusion_lm,
            args=DiffusionTrainingArguments(...),
            train_dataset=dataset,
            data_collator=RandomTruncateCollator(pad_token_id=tokenizer.pad_token_id),
        )
        trainer.train()
    """

    def __init__(self, *args, **kwargs) -> None:
        # Add NaN detection callback by default
        callbacks = kwargs.pop("callbacks", []) or []
        callbacks = list(callbacks) + [NanLossCallback()]
        super().__init__(*args, callbacks=callbacks, **kwargs)

    def compute_loss(
        self,
        model,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Compute diffusion loss by calling model forward pass.

        Args:
            model: DiffusionLM model.
            inputs: Batch dict from the data collator.
            return_outputs: If True, return (loss, outputs) tuple.

        Returns:
            loss scalar, or (loss, outputs) if return_outputs=True.
        """
        outputs = model(**inputs)
        loss = outputs["loss"]

        if return_outputs:
            return loss, outputs
        return loss
