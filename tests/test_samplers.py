"""Tests for generation samplers."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.generation import GenerationConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.models.backbone import BidirectionalTransformer
from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM
from diffusion_lm.samplers.base import SamplerOutput
from diffusion_lm.samplers.block_sampler import BlockSampler
from diffusion_lm.samplers.cached_sampler import CachedSampler
from diffusion_lm.samplers.first_hitting_sampler import FirstHittingSampler

MASK_TOKEN_ID = 999
VOCAB_SIZE = 1000
SAFE_VOCAB = MASK_TOKEN_ID


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_lm():
    """Tiny MaskedDiffusionLM for fast sampler tests."""
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.vocab_size = VOCAB_SIZE + 2
    cfg.n_embd = 64
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_positions = 128
    cfg.n_inner = 128

    import diffusion_lm.models.masked_diffusion_lm as mdlm_mod

    class _TinyBackbone(BidirectionalTransformer):
        def __init__(self, model_config):
            torch.nn.Module.__init__(self)
            self.model_config = model_config
            self.transformer = AutoModelForCausalLM.from_config(cfg)

    original = mdlm_mod.BidirectionalTransformer
    mdlm_mod.BidirectionalTransformer = _TinyBackbone

    model_config = ModelConfig(
        model_name_or_path="gpt2",
        init_from_pretrained=False,
        attn_implementation="eager",
        dtype="float32",
    )
    diffusion_config = DiffusionConfig(
        process_type="masked",
        schedule_type="linear",
        mask_token_id=MASK_TOKEN_ID,
        time_epsilon=1e-3,
    )
    lm = MaskedDiffusionLM(model_config, diffusion_config)
    mdlm_mod.BidirectionalTransformer = original
    lm.eval()
    return lm


@pytest.fixture(scope="module")
def gen_config():
    return GenerationConfig(
        max_new_tokens=8,
        num_steps=4,  # tiny for speed
        temperature=1.0,
        running_confidence_remasking=False,
    )


@pytest.fixture(scope="module")
def prompt_ids():
    B, prompt_len = 2, 4
    return torch.randint(1, SAFE_VOCAB, (B, prompt_len))


# ---------------------------------------------------------------------------
# FirstHittingSampler tests
# ---------------------------------------------------------------------------


def test_first_hitting_output_length(tiny_lm, gen_config, prompt_ids):
    """Generated sequences have correct total length (prompt + gen_len)."""
    sampler = FirstHittingSampler()
    out = sampler.generate(tiny_lm, prompt_ids, gen_config)
    assert isinstance(out, SamplerOutput)
    expected_len = prompt_ids.shape[1] + gen_config.max_new_tokens
    assert out.sequences.shape == (prompt_ids.shape[0], expected_len)


def test_prompt_preserved(tiny_lm, gen_config, prompt_ids):
    """Prompt portion of output exactly matches input prompt_ids."""
    sampler = FirstHittingSampler()
    out = sampler.generate(tiny_lm, prompt_ids, gen_config)
    prompt_len = prompt_ids.shape[1]
    assert torch.equal(out.sequences[:, :prompt_len], prompt_ids)


def test_no_mask_tokens_remain(tiny_lm, gen_config, prompt_ids):
    """No MASK tokens remain in the generated portion after sampling."""
    sampler = FirstHittingSampler()
    out = sampler.generate(tiny_lm, prompt_ids, gen_config)
    prompt_len = prompt_ids.shape[1]
    generated = out.sequences[:, prompt_len:]
    assert not (generated == MASK_TOKEN_ID).any(), "MASK tokens remain in generated output"


def test_valid_token_ids(tiny_lm, gen_config, prompt_ids):
    """All output token IDs are in [0, vocab_size)."""
    sampler = FirstHittingSampler()
    out = sampler.generate(tiny_lm, prompt_ids, gen_config)
    vocab_size = tiny_lm.backbone.transformer.config.vocab_size
    assert (out.sequences >= 0).all()
    assert (out.sequences < vocab_size).all()


def test_running_confidence_remasking(tiny_lm, prompt_ids):
    """RCR doesn't break generation (output still valid)."""
    sampler = FirstHittingSampler()
    config_rcr = GenerationConfig(
        max_new_tokens=8,
        num_steps=4,
        running_confidence_remasking=True,
    )
    out = sampler.generate(tiny_lm, prompt_ids, config_rcr)
    prompt_len = prompt_ids.shape[1]
    generated = out.sequences[:, prompt_len:]
    # No mask tokens should remain
    assert not (generated == MASK_TOKEN_ID).any()
    # Prompt still preserved
    assert torch.equal(out.sequences[:, :prompt_len], prompt_ids)


def test_sampling_produces_diverse_output(tiny_lm, prompt_ids):
    """Multinomial sampling should produce varied outputs across runs (not argmax)."""
    sampler = FirstHittingSampler()
    config = GenerationConfig(
        max_new_tokens=8,
        num_steps=4,
        temperature=1.0,
        top_p=1.0,
        running_confidence_remasking=False,
    )
    # Generate multiple times with same prompt
    outputs = []
    for _ in range(5):
        out = sampler.generate(tiny_lm, prompt_ids, config)
        outputs.append(out.sequences[:, prompt_ids.shape[1] :])

    # At least 2 out of 5 generations should differ (extremely unlikely to get
    # 5 identical outputs with multinomial sampling over 8 positions)
    n_unique = len({tuple(o[0].tolist()) for o in outputs})
    assert n_unique >= 2, (
        f"All {len(outputs)} generations produced identical output — "
        "sampler may still be using argmax instead of multinomial"
    )


def test_top_p_filtering(tiny_lm, prompt_ids):
    """Top-p filtering should not break generation."""
    sampler = FirstHittingSampler()
    config = GenerationConfig(
        max_new_tokens=8,
        num_steps=4,
        temperature=1.0,
        top_p=0.9,
        running_confidence_remasking=False,
    )
    out = sampler.generate(tiny_lm, prompt_ids, config)
    prompt_len = prompt_ids.shape[1]
    generated = out.sequences[:, prompt_len:]
    assert not (generated == MASK_TOKEN_ID).any(), "MASK tokens remain with top_p"
    assert (out.sequences >= 0).all()


# ---------------------------------------------------------------------------
# BlockSampler tests
# ---------------------------------------------------------------------------


def test_block_sampler_output_length(tiny_lm, prompt_ids):
    """BlockSampler generates correct total length."""
    sampler = BlockSampler()
    config = GenerationConfig(max_new_tokens=8, num_steps=2, block_size=4)
    out = sampler.generate(tiny_lm, prompt_ids, config)
    expected_len = prompt_ids.shape[1] + 8
    assert out.sequences.shape == (prompt_ids.shape[0], expected_len)


def test_block_sampler_no_masks_remain(tiny_lm, prompt_ids):
    """BlockSampler leaves no MASK tokens in output."""
    sampler = BlockSampler()
    config = GenerationConfig(max_new_tokens=8, num_steps=2, block_size=4)
    out = sampler.generate(tiny_lm, prompt_ids, config)
    prompt_len = prompt_ids.shape[1]
    generated = out.sequences[:, prompt_len:]
    assert not (generated == MASK_TOKEN_ID).any()


# ---------------------------------------------------------------------------
# CachedSampler tests
# ---------------------------------------------------------------------------


def test_cached_sampler_output_length(tiny_lm, prompt_ids):
    """CachedSampler generates correct total length."""
    sampler = CachedSampler()
    config = GenerationConfig(max_new_tokens=8, num_steps=4)
    out = sampler.generate(tiny_lm, prompt_ids, config)
    expected_len = prompt_ids.shape[1] + 8
    assert out.sequences.shape == (prompt_ids.shape[0], expected_len)
