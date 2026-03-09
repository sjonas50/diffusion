"""BidirectionalTransformer: wraps any HF CausalLM for bidirectional attention.

Key design decision (from research.md):
    DO NOT use _update_causal_mask monkey-patch — it is deprecated and will be
    removed in Transformers v5.10. The replacement (masking_utils.create_causal_mask)
    breaks torch.compile(fullgraph=True) (issue #42950). Silent causal fallback
    is a real failure mode (Gemma3 issue #39389).

    Instead: inject an explicit all-zeros 4D float mask (B, 1, L, L) in forward().
    All-zeros = no masking = full bidirectional attention.
    This bypasses all HF internal mask generation and is version-stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerBase

from diffusion_lm.config.model import ModelConfig


class BidirectionalTransformer(nn.Module):
    """Bidirectional attention wrapper around any HuggingFace CausalLM.

    Converts a standard causal (left-to-right) language model to bidirectional
    attention by injecting an explicit all-zeros 4D attention mask in forward().

    Supports AR-to-dLLM adaptation (recommended): start from a pretrained AR
    checkpoint (LLaMA-3, Qwen2.5, GPT-2, etc.) and fine-tune with the diffusion
    objective. Requires ~500x less compute than training from scratch.

    Args:
        config: ModelConfig specifying backbone and dtype.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model_config = config

        dtype = getattr(torch, config.dtype)
        hf_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": config.attn_implementation,
        }

        if config.init_from_pretrained:
            logger.info(
                "Loading pretrained AR checkpoint for AR-to-dLLM adaptation: "
                f"{config.model_name_or_path}"
            )
            self.transformer = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, **hf_kwargs
            )
        else:
            logger.info(
                f"Initializing model from config (random weights): {config.model_name_or_path}"
            )
            hf_config = AutoConfig.from_pretrained(config.model_name_or_path)
            self.transformer = AutoModelForCausalLM.from_config(hf_config, **hf_kwargs)

        if config.use_lora:
            self._apply_lora(config)

    def _apply_lora(self, config: ModelConfig) -> None:
        """Apply LoRA adapters for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.transformer = get_peft_model(self.transformer, lora_config)
            self.transformer.print_trainable_parameters()
        except ImportError as e:
            raise ImportError("Install peft for LoRA support: uv add peft") from e

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass with explicit bidirectional attention mask.

        Injects a 4D all-zeros float attention mask to force bidirectional
        attention, bypassing HF's internal causal mask generation.

        If attention_mask is already a 4D tensor (e.g. block-diagonal mask for
        packed sequences from the collator), it is used as-is.

        Args:
            input_ids: Token IDs, shape (B, L).
            attention_mask: Optional. If None or 2D, replaced with all-zeros 4D mask.
                If already 4D, used directly (e.g. block-diagonal for packed seqs).

        Returns:
            Logits, shape (B, L, V).
        """
        B, L = input_ids.shape
        dtype = next(self.transformer.parameters()).dtype
        device = input_ids.device

        if attention_mask is None or (attention_mask.dim() != 4):
            # All-zeros 4D float mask: no masking = full bidirectional attention.
            # In HF's attention implementation, a float mask of 0.0 means "attend"
            # (it is added to the pre-softmax scores; 0 means no masking).
            attention_mask = torch.zeros(B, 1, L, L, dtype=dtype, device=device)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs.logits

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize the token embedding table (e.g. after adding [MASK] token)."""
        self.transformer.resize_token_embeddings(new_num_tokens)


def add_mask_token(
    model: BidirectionalTransformer,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """Add [MASK] special token to tokenizer and resize model embeddings.

    Must be called before training. Sets tokenizer.mask_token and resizes
    the model's embedding table to accommodate the new token.

    Args:
        model: BidirectionalTransformer to update.
        tokenizer: Tokenizer to add [MASK] to.

    Returns:
        The new mask_token_id.
    """
    if tokenizer.mask_token is not None:
        logger.info(f"Tokenizer already has mask_token: {tokenizer.mask_token!r}")
        return tokenizer.mask_token_id

    num_added = tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(
            f"Added [MASK] token (id={tokenizer.mask_token_id}), "
            f"resized embeddings to {len(tokenizer)}"
        )
    return tokenizer.mask_token_id


def assert_bidirectional(
    model: BidirectionalTransformer,
    input_ids: Tensor | None = None,
) -> None:
    """Verify bidirectional attention is working correctly.

    Changes a token at position 5 and checks that logits at position 3 change.
    This is impossible with a causal mask (position 3 cannot attend to position 5).

    Raises:
        AssertionError: If causal attention is detected.
    """
    model.eval()
    device = next(model.parameters()).device

    if input_ids is None:
        vocab_size = model.transformer.config.vocab_size
        input_ids = torch.randint(1, vocab_size - 1, (1, 16), device=device)

    with torch.no_grad():
        logits_original = model(input_ids)

        modified = input_ids.clone()
        # Flip token at position 5 to something different
        modified[0, 5] = (modified[0, 5] + 1) % (model.transformer.config.vocab_size - 1) + 1
        logits_modified = model(modified)

    # Position 3 logits must change if attention is bidirectional
    diff = (logits_original[0, 3] - logits_modified[0, 3]).abs().max().item()
    assert diff > 1e-6, (
        f"Bidirectional attention check FAILED (diff={diff:.2e}). "
        "Causal mask may still be active."
    )
    logger.info(f"Bidirectional attention verified (max logit diff at pos 3: {diff:.4f})")
