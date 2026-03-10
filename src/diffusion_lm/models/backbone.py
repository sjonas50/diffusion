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

    For Flash Attention 2: 4D masks are not supported. Instead, we set
    ``is_causal=False`` on all attention modules and pass only a 2D padding mask.
    This matches the approach used by LLaDA and Dream.

    Supports AR-to-dLLM adaptation (recommended): start from a pretrained AR
    checkpoint (LLaMA-3, Qwen2.5, GPT-2, etc.) and fine-tune with the diffusion
    objective. Requires ~500x less compute than training from scratch.

    Args:
        config: ModelConfig specifying backbone and dtype.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model_config = config
        self._use_fa2 = config.attn_implementation == "flash_attention_2"

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

        # FA2: set is_causal=False on all attention modules for bidirectional attention.
        # FA2 does not support arbitrary 4D masks — this is the correct mechanism
        # (matches LLaDA/Dream approach, validated by HF PR #39707).
        if self._use_fa2:
            n_patched = 0
            for module in self.transformer.modules():
                if hasattr(module, "is_causal"):
                    module.is_causal = False
                    n_patched += 1
            logger.info(f"FA2 bidirectional: set is_causal=False on {n_patched} attention modules")

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
        device = input_ids.device

        if getattr(self, "_use_fa2", False):
            # FA2: pass 2D padding mask directly (or None). is_causal=False is
            # already set on attention modules in __init__. FA2 does not support
            # arbitrary 4D masks — passing one would crash or silently break.
            if attention_mask is not None and attention_mask.dim() == 4:
                # 4D mask (e.g. block-diagonal) — not supported with FA2.
                # Fall back to None (full attention). Packed sequences + FA2
                # would need flash_attn_varlen_func, which is not yet supported.
                logger.warning("4D attention mask ignored under FA2; using full attention")
                attention_mask = None
        else:
            # eager/sdpa: build explicit 4D zeros mask for bidirectional attention.
            if attention_mask is not None and attention_mask.dim() == 4:
                # Already 4D (e.g. block-diagonal for packed sequences): use as-is
                pass
            else:
                dtype = next(self.transformer.parameters()).dtype
                mask_4d = torch.zeros(B, 1, L, L, dtype=dtype, device=device)
                if attention_mask is not None and attention_mask.dim() == 2:
                    # attention_mask: (B, L), 1=real, 0=padding
                    # Expand to (B, 1, 1, L) so padding KEYS get -inf for all queries
                    pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
                    mask_4d = mask_4d + pad_mask * torch.finfo(dtype).min
                attention_mask = mask_4d

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
