"""Shared pytest fixtures for diffusion-lm tests."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return CPU device for tests (no GPU required)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tiny_gpt2_config() -> GPT2Config:
    """Tiny GPT-2 config for fast model instantiation in tests."""
    return GPT2Config(
        vocab_size=1000,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_inner=128,
    )


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """GPT-2 tokenizer with added [MASK] token."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "[MASK]", "pad_token": "[PAD]"})
    return tokenizer


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def seq_len() -> int:
    return 32
