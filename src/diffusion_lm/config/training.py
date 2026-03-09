"""Training arguments extending HuggingFace TrainingArguments."""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class DiffusionTrainingArguments(TrainingArguments):
    """HF TrainingArguments extended with diffusion-specific hyperparameters.

    Inherits all standard HF Trainer arguments (lr, batch size, etc.) and adds:

    Args:
        random_truncation_ratio: Fraction of batches to randomly truncate during
            pretraining (LLaDA technique for robustness to variable lengths).
        dpo_beta: DPO temperature controlling deviation from reference policy.
        dpo_num_mc_samples: Monte Carlo samples for ELBO log-prob estimation in DPO.
            Higher = lower variance, more compute. 8 is a good default.
        grpo_group_size: Number of completions sampled per prompt in GRPO.
        grpo_clip_ratio: PPO-style clipping ratio epsilon for GRPO.
        grpo_num_mc_samples: Monte Carlo samples for ELBO log-prob estimation in GRPO.
    """

    random_truncation_ratio: float = field(
        default=0.01,
        metadata={"help": "Fraction of batches to randomly truncate (LLaDA technique)."},
    )
    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "DPO temperature beta."},
    )
    dpo_num_mc_samples: int = field(
        default=8,
        metadata={"help": "Monte Carlo samples for ELBO log-prob estimation in DPO."},
    )
    grpo_group_size: int = field(
        default=8,
        metadata={"help": "Number of completions per prompt in GRPO."},
    )
    grpo_clip_ratio: float = field(
        default=0.2,
        metadata={"help": "PPO-style clip ratio epsilon for GRPO."},
    )
    grpo_num_mc_samples: int = field(
        default=8,
        metadata={"help": "Monte Carlo samples for ELBO log-prob estimation in GRPO."},
    )
