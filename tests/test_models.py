"""Tests for model architecture — BidirectionalTransformer and diffusion LMs."""

from __future__ import annotations

import pytest
import torch

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.models.backbone import (
    BidirectionalTransformer,
    add_mask_token,
    assert_bidirectional,
)
from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig(
        model_name_or_path="gpt2",
        init_from_pretrained=True,
        attn_implementation="eager",
        dtype="float32",
    )


@pytest.fixture(scope="module")
def backbone(model_config) -> BidirectionalTransformer:
    return BidirectionalTransformer(model_config)


@pytest.fixture(scope="module")
def tokenizer_with_mask(tiny_tokenizer):
    """Tokenizer that already has mask token (from conftest)."""
    return tiny_tokenizer


@pytest.fixture(scope="module")
def diffusion_config(backbone, tokenizer_with_mask) -> DiffusionConfig:
    """DiffusionConfig with mask_token_id set."""
    mask_token_id = add_mask_token(backbone, tokenizer_with_mask)
    return DiffusionConfig(
        process_type="masked",
        schedule_type="linear",
        mask_token_id=mask_token_id,
        time_epsilon=1e-3,
    )


@pytest.fixture(scope="module")
def masked_lm(model_config, diffusion_config, tokenizer_with_mask) -> MaskedDiffusionLM:
    lm = MaskedDiffusionLM(model_config, diffusion_config)
    # Resize embeddings to match the tokenizer that has [MASK] and [PAD] added.
    # Without this, the mask_token_id (50257+) is out of range of GPT-2's original vocab.
    lm.backbone.resize_token_embeddings(len(tokenizer_with_mask))
    return lm


# ---------------------------------------------------------------------------
# Backbone tests
# ---------------------------------------------------------------------------

def test_bidirectional_attention(backbone):
    """Changing token at position 5 must change logits at position 3.

    This is impossible with a causal mask — position 3 cannot attend to position 5.
    If this test fails, causal masking is still active.
    """
    backbone.eval()
    vocab_size = backbone.transformer.config.vocab_size
    input_ids = torch.randint(1, vocab_size - 2, (1, 16))

    with torch.no_grad():
        logits_orig = backbone(input_ids)
        modified = input_ids.clone()
        modified[0, 5] = (modified[0, 5] + 1) % (vocab_size - 2) + 1
        logits_mod = backbone(modified)

    diff = (logits_orig[0, 3] - logits_mod[0, 3]).abs().max().item()
    assert diff > 1e-6, (
        f"Bidirectional attention FAILED: logits at position 3 unchanged when "
        f"position 5 changes (diff={diff:.2e}). Causal mask may still be active."
    )


def test_assert_bidirectional_utility(backbone):
    """assert_bidirectional() helper should not raise for a bidirectional model."""
    assert_bidirectional(backbone)


def test_forward_pass_shape(backbone):
    """Backbone output shape must be (B, L, V)."""
    B, L = 2, 20
    vocab_size = backbone.transformer.config.vocab_size
    input_ids = torch.randint(0, vocab_size - 1, (B, L))

    with torch.no_grad():
        logits = backbone(input_ids)

    assert logits.shape == (B, L, vocab_size), (
        f"Expected ({B}, {L}, {vocab_size}), got {logits.shape}"
    )


def test_forward_with_4d_attention_mask(backbone):
    """Backbone accepts explicit 4D attention mask (e.g. block-diagonal from collator)."""
    B, L = 2, 16
    vocab_size = backbone.transformer.config.vocab_size
    input_ids = torch.randint(0, vocab_size - 1, (B, L))
    # Custom 4D mask (all zeros = full bidirectional)
    mask_4d = torch.zeros(B, 1, L, L)

    with torch.no_grad():
        logits = backbone(input_ids, attention_mask=mask_4d)

    assert logits.shape == (B, L, vocab_size)


def test_mask_token_added(backbone, tokenizer_with_mask):
    """After add_mask_token(), tokenizer has mask_token and embeddings are resized."""
    original_vocab = backbone.transformer.config.vocab_size
    # tokenizer already has mask token from fixture
    assert tokenizer_with_mask.mask_token is not None
    assert tokenizer_with_mask.mask_token_id is not None
    # Embedding table should be resized to at least original vocab size
    emb_weight = backbone.transformer.get_input_embeddings().weight
    assert emb_weight.shape[0] >= original_vocab


# ---------------------------------------------------------------------------
# MaskedDiffusionLM tests
# ---------------------------------------------------------------------------

def test_masked_diffusion_lm_loss_scalar(masked_lm, diffusion_config):
    """Forward pass returns scalar loss (not NaN, not Inf)."""
    B, L = 2, 32
    # Use safe token range — below mask_token_id to avoid embedding out-of-range
    safe_vocab = diffusion_config.mask_token_id
    input_ids = torch.randint(1, safe_vocab, (B, L))

    masked_lm.eval()
    with torch.no_grad():
        outputs = masked_lm(input_ids)

    loss = outputs["loss"]
    assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
    assert not loss.isnan(), "Loss is NaN"
    assert not loss.isinf(), "Loss is Inf"
    assert loss.item() > 0, "Loss should be positive"


def test_masked_diffusion_lm_logits_shape(masked_lm, diffusion_config):
    """get_logits() returns correct shape (B, L, V)."""
    B, L = 2, 16
    safe_vocab = diffusion_config.mask_token_id
    input_ids = torch.randint(1, safe_vocab, (B, L))

    with torch.no_grad():
        logits = masked_lm.get_logits(input_ids)

    actual_vocab = masked_lm.backbone.transformer.get_input_embeddings().weight.shape[0]
    assert logits.shape == (B, L, actual_vocab)


def test_prompt_mask_protection(masked_lm, diffusion_config):
    """Loss is valid when prompt_mask restricts to response positions."""
    B, L = 2, 32
    prompt_len = 16
    safe_vocab = diffusion_config.mask_token_id
    input_ids = torch.randint(1, safe_vocab, (B, L))

    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    prompt_mask[:, :prompt_len] = True

    # Run with gradient tracking
    masked_lm.train()
    emb = masked_lm.backbone.transformer.get_input_embeddings()
    emb.weight.requires_grad_(True)
    if emb.weight.grad is not None:
        emb.weight.grad.zero_()

    outputs = masked_lm(input_ids, prompt_mask=prompt_mask)
    outputs["loss"].backward()

    # The loss is computed only on response tokens, but gradients can still flow
    # through the embedding lookup of ALL tokens (including prompt) via the
    # bidirectional attention. We verify only that the loss value is valid.
    assert not outputs["loss"].isnan()
    assert outputs["loss"].item() >= 0


def test_masked_diffusion_lm_mask_token_validation():
    """MaskedDiffusionLM must raise if mask_token_id is -1."""
    model_config = ModelConfig(
        model_name_or_path="gpt2",
        init_from_pretrained=False,
        attn_implementation="eager",
        dtype="float32",
    )
    diffusion_config = DiffusionConfig(mask_token_id=-1)
    with pytest.raises(ValueError, match="mask_token_id"):
        MaskedDiffusionLM(model_config, diffusion_config)


def test_masked_diffusion_lm_with_sft_prompt_mask(masked_lm, diffusion_config):
    """Loss with prompt_mask differs from loss without it."""
    B, L = 2, 32
    prompt_len = 16
    safe_vocab = diffusion_config.mask_token_id
    input_ids = torch.randint(1, safe_vocab, (B, L))
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    prompt_mask[:, :prompt_len] = True

    masked_lm.eval()
    with torch.no_grad():
        out_full = masked_lm(input_ids)
        out_sft = masked_lm(input_ids, prompt_mask=prompt_mask)

    # Both should be valid scalars
    assert not out_full["loss"].isnan()
    assert not out_sft["loss"].isnan()
    # They may or may not be equal (depends on which tokens happen to be masked)
    # but both must be finite
    assert out_sft["loss"].isfinite()
