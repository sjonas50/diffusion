"""Tests for diffusion trainers."""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.data.collators import RandomTruncateCollator, SFTCollator
from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM
from diffusion_lm.trainers.base import NanLossCallback
from diffusion_lm.trainers.sft import SFTTrainer

MASK_TOKEN_ID = 999
VOCAB_SIZE = 1000
SAFE_VOCAB = MASK_TOKEN_ID  # token ids in [0, MASK_TOKEN_ID)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_config():
    return ModelConfig(
        model_name_or_path="gpt2",
        init_from_pretrained=False,  # Random weights for speed
        attn_implementation="eager",
        dtype="float32",
    )


@pytest.fixture(scope="module")
def tiny_diffusion_config():
    return DiffusionConfig(
        process_type="masked",
        schedule_type="linear",
        mask_token_id=MASK_TOKEN_ID,
        time_epsilon=1e-3,
    )


@pytest.fixture(scope="module")
def tiny_masked_lm(tiny_model_config, tiny_diffusion_config):
    """Tiny MaskedDiffusionLM backed by a tiny GPT-2 (random weights)."""
    # Patch GPT-2 config to be tiny for fast tests
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.vocab_size = VOCAB_SIZE + 2  # room for mask + pad
    cfg.n_embd = 64
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_positions = 64
    cfg.n_inner = 128

    from transformers import AutoModelForCausalLM

    from diffusion_lm.models.backbone import BidirectionalTransformer

    class _TinyBackbone(BidirectionalTransformer):
        def __init__(self, model_config):
            torch.nn.Module.__init__(self)
            self.model_config = model_config
            self.transformer = AutoModelForCausalLM.from_config(cfg)

    import diffusion_lm.models.masked_diffusion_lm as mdlm_mod
    original_backbone = mdlm_mod.BidirectionalTransformer

    # Monkey-patch to use tiny backbone
    mdlm_mod.BidirectionalTransformer = _TinyBackbone
    lm = MaskedDiffusionLM(tiny_model_config, tiny_diffusion_config)
    mdlm_mod.BidirectionalTransformer = original_backbone
    return lm


@pytest.fixture
def simple_batch(tiny_diffusion_config):
    """Small batch of pretraining data."""
    B, L = 2, 16
    return {
        "input_ids": torch.randint(1, SAFE_VOCAB, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
    }


@pytest.fixture
def sft_batch(tiny_diffusion_config):
    """Small SFT batch with prompt_mask."""
    B, L = 2, 16
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    prompt_mask[:, :8] = True  # first 8 tokens = prompt
    return {
        "input_ids": torch.randint(1, SAFE_VOCAB, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "prompt_mask": prompt_mask,
    }


# ---------------------------------------------------------------------------
# DiffusionTrainer (base) tests
# ---------------------------------------------------------------------------

def test_pretrain_one_step(tiny_masked_lm, simple_batch):
    """One optimizer step completes without error on random data."""
    tiny_masked_lm.train()
    optimizer = AdamW(tiny_masked_lm.parameters(), lr=1e-4)

    outputs = tiny_masked_lm(**simple_batch)
    loss = outputs["loss"]

    assert not loss.isnan(), "Loss is NaN on first step"
    assert loss.item() > 0

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_sft_one_step(tiny_masked_lm, sft_batch):
    """SFT step with prompt_mask — loss should be valid."""
    tiny_masked_lm.train()
    optimizer = AdamW(tiny_masked_lm.parameters(), lr=1e-4)

    outputs = tiny_masked_lm(**sft_batch)
    loss = outputs["loss"]

    assert not loss.isnan()
    assert loss.item() >= 0

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_loss_decreases(tiny_masked_lm, simple_batch):
    """Over 20 gradient steps, average loss over last 5 steps < average over first 5 steps.

    Diffusion training is stochastic (random masking each step), so we compare
    moving averages rather than step-by-step values.
    """
    tiny_masked_lm.train()
    # High LR to force obvious overfitting on a tiny fixed batch
    optimizer = AdamW(tiny_masked_lm.parameters(), lr=5e-3)

    losses = []
    for _ in range(40):
        outputs = tiny_masked_lm(**simple_batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10
    assert last_avg < first_avg, (
        f"Loss did not decrease on average: first_10_avg={first_avg:.4f}, "
        f"last_10_avg={last_avg:.4f}. "
        f"Full losses: {[f'{v:.2f}' for v in losses]}"
    )


def test_nan_loss_callback():
    """NanLossCallback stops training when NaN loss is detected."""
    from transformers import TrainerControl, TrainerState, TrainingArguments

    callback = NanLossCallback()
    args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
    state = TrainerState()
    state.global_step = 42
    control = TrainerControl()

    callback.on_log(args, state, control, logs={"loss": float("nan")})
    assert control.should_training_stop, "Callback should stop training on NaN loss"


def test_nan_loss_callback_normal_loss():
    """NanLossCallback does NOT stop training on normal loss."""
    from transformers import TrainerControl, TrainerState, TrainingArguments

    callback = NanLossCallback()
    args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
    state = TrainerState()
    control = TrainerControl()

    callback.on_log(args, state, control, logs={"loss": 2.345})
    assert not control.should_training_stop


def test_sft_trainer_requires_prompt_mask(tiny_masked_lm, simple_batch):
    """SFTTrainer raises ValueError if prompt_mask is missing from batch."""

    trainer = SFTTrainer(model=tiny_masked_lm, args=None, pad_token_id=0)

    with pytest.raises(ValueError, match="prompt_mask"):
        trainer.compute_loss(tiny_masked_lm, simple_batch)


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------

def test_random_truncation_collator():
    """RandomTruncateCollator truncates with approximately correct probability."""

    collator = RandomTruncateCollator(pad_token_id=0, truncation_ratio=1.0, min_length=4)

    # With ratio=1.0, every call should truncate
    batch = [{"input_ids": list(range(32))} for _ in range(4)]
    result = collator(batch)
    # Truncated length should be < 32
    assert result["input_ids"].shape[1] <= 32


def test_random_truncation_collator_no_truncation():
    """With ratio=0.0, no truncation occurs."""
    collator = RandomTruncateCollator(pad_token_id=0, truncation_ratio=0.0)
    batch = [{"input_ids": list(range(16))} for _ in range(4)]
    result = collator(batch)
    assert result["input_ids"].shape[1] == 16


def test_sft_collator_pads_prompt_mask():
    """SFTCollator aligns prompt_mask padding with input_ids padding."""
    collator = SFTCollator(pad_token_id=0)
    batch = [
        {"input_ids": [1, 2, 3, 4], "prompt_mask": [True, True, False, False]},
        {"input_ids": [5, 6], "prompt_mask": [True, False]},
    ]
    result = collator(batch)
    assert result["input_ids"].shape == result["prompt_mask"].shape
    # Shorter sequence padded to match longer
    assert result["prompt_mask"].shape == (2, 4)
    # Padded positions should be False
    assert not result["prompt_mask"][1, 2].item()
    assert not result["prompt_mask"][1, 3].item()


def test_packed_sequence_block_diagonal_mask():
    """RandomTruncateCollator with pack_sequences=True emits block-diagonal mask."""
    collator = RandomTruncateCollator(
        pad_token_id=0, truncation_ratio=0.0, pack_sequences=True
    )
    batch = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6]},
    ]
    result = collator(batch)
    # attention_mask should be 4D when pack_sequences=True
    assert result["attention_mask"].dim() == 4, (
        "Expected 4D block-diagonal mask when pack_sequences=True"
    )
