"""Preference data pipeline for DPO/GRPO training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class PreferenceDataset:
    """Dataset for preference-based training (DPO/GRPO).

    Format: {"prompt": str, "chosen": str, "rejected": str}

    Produces:
        chosen_input_ids: prompt + chosen response tokens.
        rejected_input_ids: prompt + rejected response tokens.
        prompt_mask: True at prompt positions, False at response positions.

    Args:
        data: List of preference examples.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum total sequence length.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._process(ex) for ex in data]
        logger.info(f"PreferenceDataset: {len(self.examples)} examples")

    def _process(self, example: dict) -> dict[str, list]:
        prompt_ids = self.tokenizer.encode(example["prompt"], add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(example["chosen"], add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(example["rejected"], add_special_tokens=False)

        chosen_input_ids = (prompt_ids + chosen_ids)[: self.max_length]
        rejected_input_ids = (prompt_ids + rejected_ids)[: self.max_length]

        prompt_len = min(len(prompt_ids), self.max_length)
        # prompt_mask aligned to chosen (the shorter of the two, prompt portion is the same)
        prompt_mask = [True] * prompt_len + [False] * (len(chosen_input_ids) - prompt_len)

        return {
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "prompt_mask": prompt_mask,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list]:
        return self.examples[idx]
