"""Configuration dataclasses for diffusion-lm."""

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.generation import GenerationConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.config.training import DiffusionTrainingArguments

__all__ = [
    "DiffusionConfig",
    "DiffusionTrainingArguments",
    "GenerationConfig",
    "ModelConfig",
]
