"""Integration tests verifying bidirectional attention works across model families.

These tests load real pretrained weights and are marked `slow` — they are
skipped in normal CI and run explicitly via:

    uv run pytest tests/test_model_compat.py -v -m slow

Each test verifies:
1. Model loads without error.
2. Bidirectionality: changing a token at position 5 changes logits at position 3.
3. [MASK] token can be added and embeddings resized.
4. A full forward pass produces valid (non-NaN) loss.
5. FirstHittingSampler generates a sequence with no remaining MASK tokens.
"""

from __future__ import annotations

import pytest
import torch

# Models to test: (pytest_id, hf_model_id, needs_hf_token)
MODEL_CASES = [
    ("qwen2-5-0-5b", "Qwen/Qwen2.5-0.5B", False),
    ("qwen3-0-6b", "Qwen/Qwen3-0.6B", False),
    ("llama-3-2-1b", "meta-llama/Llama-3.2-1B", True),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_model(model_id: str):
    """Build MaskedDiffusionLM from a real pretrained checkpoint."""
    from transformers import AutoTokenizer

    from diffusion_lm.config.diffusion import DiffusionConfig
    from diffusion_lm.config.model import ModelConfig
    from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token
    from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_config = ModelConfig(
        model_name_or_path=model_id,
        init_from_pretrained=True,
        attn_implementation="eager",
        dtype="float32",
    )
    backbone = BidirectionalTransformer(model_config)
    mask_token_id = add_mask_token(backbone, tokenizer)

    diffusion_config = DiffusionConfig(
        process_type="masked",
        schedule_type="linear",
        mask_token_id=mask_token_id,
        time_epsilon=1e-3,
    )
    model = MaskedDiffusionLM(model_config, diffusion_config)
    model.backbone = backbone
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_id,needs_token", [(m, t) for _, m, t in MODEL_CASES],
                         ids=[i for i, _, _ in MODEL_CASES])
def test_bidirectionality(model_id, needs_token):
    """Logits at position 3 change when token at position 5 changes."""
    pytest.importorskip("transformers")

    model, tokenizer = _build_model(model_id)
    vocab_size = model.backbone.transformer.config.vocab_size
    B, L = 1, 16

    x1 = torch.randint(1, min(vocab_size - 2, 50000), (B, L))
    x2 = x1.clone()
    x2[0, 5] = (x2[0, 5] + 1) % (vocab_size - 2) + 1  # change token at pos 5

    prompt_mask = torch.zeros(B, L, dtype=torch.bool)

    with torch.no_grad():
        logits1_full = model.get_logits(x1)
        logits2_full = model.get_logits(x2)

    logits1 = logits1_full[0, 3]
    logits2 = logits2_full[0, 3]

    assert not torch.allclose(logits1, logits2, atol=1e-4), (
        f"{model_id}: logits at pos 3 did NOT change when pos 5 changed — "
        "model appears to still be causal!"
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_id,needs_token", [(m, t) for _, m, t in MODEL_CASES],
                         ids=[i for i, _, _ in MODEL_CASES])
def test_forward_no_nan(model_id, needs_token):
    """Forward pass produces a finite, positive loss."""
    model, tokenizer = _build_model(model_id)
    vocab_size = model.backbone.transformer.config.vocab_size
    B, L = 2, 32

    input_ids = torch.randint(1, min(vocab_size - 2, 50000), (B, L))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    loss = outputs["loss"]
    assert not loss.isnan(), f"{model_id}: loss is NaN"
    assert not loss.isinf(), f"{model_id}: loss is Inf"
    assert loss.item() > 0, f"{model_id}: loss is non-positive ({loss.item():.4f})"


@pytest.mark.slow
@pytest.mark.parametrize("model_id,needs_token", [(m, t) for _, m, t in MODEL_CASES],
                         ids=[i for i, _, _ in MODEL_CASES])
def test_generation_no_mask_tokens_remain(model_id, needs_token):
    """FirstHittingSampler leaves no [MASK] tokens in the output."""
    from diffusion_lm.config.generation import GenerationConfig
    from diffusion_lm.samplers.first_hitting_sampler import FirstHittingSampler

    model, tokenizer = _build_model(model_id)
    mask_token_id = model.diffusion.mask_token_id
    vocab_size = model.backbone.transformer.config.vocab_size

    prompt = torch.randint(1, min(vocab_size - 2, 50000), (1, 8))
    gen_config = GenerationConfig(
        max_new_tokens=16, num_steps=8, running_confidence_remasking=False
    )

    sampler = FirstHittingSampler()
    with torch.no_grad():
        out = sampler.generate(model, prompt, gen_config)

    generated = out.sequences[:, prompt.shape[1]:]
    assert not (generated == mask_token_id).any(), (
        f"{model_id}: MASK tokens remain in generated output"
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_id,needs_token", [(m, t) for _, m, t in MODEL_CASES],
                         ids=[i for i, _, _ in MODEL_CASES])
def test_mask_token_added(model_id, needs_token):
    """[MASK] token is added to tokenizer and embeddings are resized correctly."""
    from transformers import AutoTokenizer

    from diffusion_lm.config.model import ModelConfig
    from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    original_vocab_size = len(tokenizer)

    model_config = ModelConfig(
        model_name_or_path=model_id,
        init_from_pretrained=False,  # random weights for speed
        attn_implementation="eager",
        dtype="float32",
    )
    backbone = BidirectionalTransformer(model_config)

    if tokenizer.mask_token is None:
        mask_token_id = add_mask_token(backbone, tokenizer)
        assert len(tokenizer) == original_vocab_size + 1
        assert mask_token_id == original_vocab_size
    else:
        # Model already has a mask token (e.g. BERT-style)
        mask_token_id = tokenizer.mask_token_id
        assert mask_token_id is not None

    # Verify embedding table matches tokenizer
    embed_size = backbone.transformer.get_input_embeddings().weight.shape[0]
    assert embed_size == len(tokenizer), (
        f"{model_id}: embedding size {embed_size} != tokenizer vocab {len(tokenizer)}"
    )
