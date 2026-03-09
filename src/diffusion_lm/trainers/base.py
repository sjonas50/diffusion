"""Base diffusion trainer extending HuggingFace Trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """Save model, breaking tied weights for safetensors compatibility.

        GPT-2, Qwen, and LLaMA share embed_tokens.weight with lm_head.weight.
        safetensors rejects shared tensors. We temporarily detach lm_head to
        produce an independent copy for saving, then restore the tie.
        """
        lm_head = None
        old_weight = None

        # Detect tied lm_head inside our backbone wrapper
        try:
            transformer = self.model.backbone.transformer
            lm_head = getattr(transformer, "lm_head", None)
            if lm_head is not None:
                embed = transformer.get_input_embeddings()
                if lm_head.weight.data_ptr() == embed.weight.data_ptr():
                    old_weight = lm_head.weight
                    lm_head.weight = torch.nn.Parameter(old_weight.detach().clone())
                    logger.debug("Temporarily broke lm_head/embed_tokens weight tie for saving")
        except AttributeError:
            pass

        try:
            super()._save(output_dir, state_dict)
        finally:
            # Restore the tie regardless of success/failure
            if lm_head is not None and old_weight is not None:
                lm_head.weight = old_weight

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
