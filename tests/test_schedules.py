"""Tests for noise schedules."""

from __future__ import annotations

import pytest
import torch

from diffusion_lm.schedules import CosineSchedule, LinearSchedule, LogLinearSchedule
from diffusion_lm.schedules.base import NoiseSchedule

ALL_SCHEDULES = [LinearSchedule(), CosineSchedule(), LogLinearSchedule()]
SCHEDULE_IDS = ["linear", "cosine", "loglinear"]


@pytest.mark.parametrize("schedule", ALL_SCHEDULES, ids=SCHEDULE_IDS)
def test_alpha_boundary_conditions(schedule: NoiseSchedule):
    """alpha(0) ≈ 1 (fully clean), alpha(1) ≈ 0 (fully corrupted)."""
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    assert schedule.alpha(t0).item() == pytest.approx(1.0, abs=1e-5), "alpha(0) must be 1"
    assert schedule.alpha(t1).item() == pytest.approx(0.0, abs=1e-3), "alpha(1) must be ~0"


@pytest.mark.parametrize("schedule", ALL_SCHEDULES, ids=SCHEDULE_IDS)
def test_alpha_monotone_decreasing(schedule: NoiseSchedule):
    """alpha(t) is strictly decreasing."""
    t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    alphas = schedule.alpha(t)
    diffs = alphas[1:] - alphas[:-1]
    assert (diffs < 0).all(), f"alpha not monotone decreasing: {alphas.tolist()}"


@pytest.mark.parametrize("schedule", ALL_SCHEDULES, ids=SCHEDULE_IDS)
def test_alpha_in_unit_interval(schedule: NoiseSchedule):
    """alpha(t) ∈ [0, 1] for all t ∈ [0, 1]."""
    t = torch.linspace(0.0, 1.0, 100)
    alphas = schedule.alpha(t)
    assert (alphas >= 0).all(), "alpha must be >= 0"
    assert (alphas <= 1).all(), "alpha must be <= 1"


@pytest.mark.parametrize("schedule", ALL_SCHEDULES, ids=SCHEDULE_IDS)
def test_weight_positive(schedule: NoiseSchedule):
    """Loss weight w(t) = -alpha'(t)/(1-alpha(t)) > 0 for t ∈ (0, 1)."""
    t = torch.linspace(0.01, 0.99, 50)
    w = schedule.weight(t)
    assert (w > 0).all(), f"weight must be positive, got min={w.min().item():.6f}"


@pytest.mark.parametrize("schedule", ALL_SCHEDULES, ids=SCHEDULE_IDS)
def test_mask_probability_complement(schedule: NoiseSchedule):
    """mask_probability(t) = 1 - alpha(t)."""
    t = torch.linspace(0.1, 0.9, 20)
    assert torch.allclose(
        schedule.mask_probability(t),
        1.0 - schedule.alpha(t),
        atol=1e-6,
    )


@pytest.mark.parametrize("schedule", ALL_SCHEDULES, ids=SCHEDULE_IDS)
def test_alpha_accepts_batched_input(schedule: NoiseSchedule):
    """alpha() handles arbitrary tensor shapes."""
    t = torch.rand(4, 3, 2)
    alphas = schedule.alpha(t)
    assert alphas.shape == t.shape
