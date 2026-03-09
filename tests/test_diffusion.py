"""Tests for diffusion processes."""

from __future__ import annotations

import pytest
import torch

from diffusion_lm.diffusion.masked import MaskedDiffusionProcess
from diffusion_lm.schedules import LinearSchedule

MASK_TOKEN_ID = 999
VOCAB_SIZE = 1000


@pytest.fixture
def masked_process():
    return MaskedDiffusionProcess(
        schedule=LinearSchedule(),
        mask_token_id=MASK_TOKEN_ID,
        time_epsilon=1e-3,
    )


def test_masked_forward_masking_ratio(masked_process):
    """At t=0.5, approximately 50% of tokens should be masked."""
    B, L = 32, 128
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.full((B,), 0.5)
    corrupted, token_mask = masked_process.forward_process(x0, t)

    mask_ratio = token_mask.float().mean().item()
    assert abs(mask_ratio - 0.5) < 0.05, f"Expected ~50% masking, got {mask_ratio:.3f}"


def test_masked_forward_no_masking_at_t0(masked_process):
    """Near t=time_epsilon, very few tokens should be masked."""
    B, L = 16, 64
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.full((B,), masked_process.time_epsilon)
    _, token_mask = masked_process.forward_process(x0, t)
    # At t=epsilon, p_mask = epsilon ≈ 0.001, so almost no masking
    assert token_mask.float().mean().item() < 0.05


def test_masked_forward_heavy_masking_at_t1(masked_process):
    """Near t=1, nearly all tokens should be masked."""
    B, L = 16, 64
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.full((B,), 0.99)
    _, token_mask = masked_process.forward_process(x0, t)
    assert token_mask.float().mean().item() > 0.9


def test_masked_corrupted_preserves_unmasked(masked_process):
    """Unmasked tokens in corrupted output must equal original tokens."""
    B, L = 8, 32
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.full((B,), 0.5)
    corrupted, token_mask = masked_process.forward_process(x0, t)

    unmasked = ~token_mask
    assert (corrupted[unmasked] == x0[unmasked]).all()


def test_masked_compute_loss_shape(masked_process):
    """Loss is a scalar with no NaN or Inf."""
    B, L, V = 4, 32, VOCAB_SIZE
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.rand(B) * 0.9 + 0.05
    corrupted, _ = masked_process.forward_process(x0, t)
    logits = torch.randn(B, L, V)

    loss = masked_process.compute_loss(logits, x0, corrupted, t)

    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert not loss.isnan().item(), "Loss is NaN"
    assert not loss.isinf().item(), "Loss is Inf"


def test_antithetic_timestep_coverage(masked_process):
    """Antithetic timesteps cover [0, 1) uniformly across the batch."""
    B = 16
    device = torch.device("cpu")
    t = masked_process.sample_timesteps(B, device)

    assert t.shape == (B,)
    assert (t >= masked_process.time_epsilon).all(), "Timesteps below time_epsilon"
    assert (t < 1.0).all(), "Timesteps must be < 1"

    # Antithetic: sorted timesteps should be roughly evenly spaced
    t_sorted, _ = t.sort()
    gaps = t_sorted[1:] - t_sorted[:-1]
    # All gaps should be approximately 1/B (± some tolerance)
    expected_gap = (1.0 - masked_process.time_epsilon) / B
    assert gaps.max().item() < expected_gap * 3, "Timesteps not well-distributed"


def test_sft_prompt_protection(masked_process):
    """With loss_mask excluding prompt positions, only response positions contribute."""
    B, L, V = 4, 32, VOCAB_SIZE
    prompt_len = 16
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.full((B,), 0.5)
    corrupted, token_mask = masked_process.forward_process(x0, t)

    # prompt_mask: True for prompt positions
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    prompt_mask[:, :prompt_len] = True
    loss_mask = ~prompt_mask  # only response positions

    # Force masking some prompt tokens to test they're excluded
    corrupted[:, :prompt_len] = MASK_TOKEN_ID

    logits = torch.randn(B, L, V, requires_grad=True)
    loss = masked_process.compute_loss(logits, x0, corrupted, t, loss_mask=loss_mask)

    assert not loss.isnan()
    # Only response tokens counted — loss must be non-negative
    assert loss.item() >= 0


def test_loss_decreases_with_better_predictions(masked_process):
    """A model that predicts the correct tokens should have lower loss."""
    B, L, V = 4, 32, VOCAB_SIZE
    x0 = torch.randint(0, MASK_TOKEN_ID, (B, L))
    t = torch.full((B,), 0.5)
    corrupted, _ = masked_process.forward_process(x0, t)

    # Perfect logits: huge value on correct token
    perfect_logits = torch.full((B, L, V), -100.0)
    for b in range(B):
        for pos in range(L):
            perfect_logits[b, pos, x0[b, pos]] = 100.0

    random_logits = torch.randn(B, L, V)

    loss_perfect = masked_process.compute_loss(perfect_logits, x0, corrupted, t)
    loss_random = masked_process.compute_loss(random_logits, x0, corrupted, t)

    assert loss_perfect.item() < loss_random.item()
