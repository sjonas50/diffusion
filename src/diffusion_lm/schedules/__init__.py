"""Noise schedules for diffusion-lm."""

from diffusion_lm.schedules.base import NoiseSchedule
from diffusion_lm.schedules.cosine import CosineSchedule
from diffusion_lm.schedules.linear import LinearSchedule
from diffusion_lm.schedules.loglinear import LogLinearSchedule

__all__ = ["CosineSchedule", "LinearSchedule", "LogLinearSchedule", "NoiseSchedule"]

SCHEDULE_REGISTRY: dict[str, type[NoiseSchedule]] = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "loglinear": LogLinearSchedule,
}


def build_schedule(schedule_type: str) -> NoiseSchedule:
    """Instantiate a noise schedule by name."""
    if schedule_type not in SCHEDULE_REGISTRY:
        valid = list(SCHEDULE_REGISTRY)
        raise ValueError(f"Unknown schedule '{schedule_type}'. Choose from {valid}")
    return SCHEDULE_REGISTRY[schedule_type]()
