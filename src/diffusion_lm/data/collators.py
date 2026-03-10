"""Data collators for diffusion LM training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _pad_sequence(sequences: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """Pad a list of token ID sequences to the same length.

    Returns:
        input_ids: (B, L_max) padded tensor.
        attention_mask: (B, L_max) 1 for real tokens, 0 for padding.
    """
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def make_block_diagonal_mask(lengths: list[int], max_len: int) -> Tensor:
    """Build a 4D block-diagonal attention mask for packed sequences.

    When packing multiple sequences into one row, standard attention would allow
    tokens from different sequences to attend to each other — silent quality
    degradation with no crash signal.

    This mask restricts attention to within each packed sequence.

    Args:
        lengths: Length of each packed sequence segment.
        max_len: Total padded sequence length.

    Returns:
        Float mask of shape (1, 1, max_len, max_len).
        0.0 = attend, -inf = do not attend (added to pre-softmax scores).
    """
    mask = torch.full((max_len, max_len), float("-inf"))
    start = 0
    for length in lengths:
        end = start + length
        mask[start:end, start:end] = 0.0
        start = end
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, max_len)


@dataclass
class RandomTruncateCollator:
    """Collator for pretraining that optionally truncates batches randomly.

    LLaDA technique: with probability `truncation_ratio`, truncates the batch
    to a random length in [min_length, max_length]. Improves robustness to
    variable-length sequences at inference.

    Emits per-sample block-diagonal attention mask when `pack_sequences=True`.

    Args:
        pad_token_id: Token ID for padding.
        truncation_ratio: Probability of random truncation (default 0.01 = 1%).
        min_length: Minimum truncation length.
        pack_sequences: If True, emit 4D block-diagonal masks for packed sequences.
    """

    pad_token_id: int
    truncation_ratio: float = 0.01
    min_length: int = 16
    pack_sequences: bool = False

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        sequences = [item["input_ids"] for item in batch]
        input_ids, attention_mask = _pad_sequence(sequences, self.pad_token_id)
        max_len = input_ids.shape[1]

        # Random truncation (LLaDA technique)
        if random.random() < self.truncation_ratio:
            trunc_len = random.randint(self.min_length, max_len)
            input_ids = input_ids[:, :trunc_len]
            attention_mask = attention_mask[:, :trunc_len]

        result: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.pack_sequences:
            current_len = input_ids.shape[1]
            # Use truncated lengths (clamped to current sequence length)
            lengths = [min(len(s), current_len) for s in sequences]
            result["attention_mask"] = make_block_diagonal_mask(lengths, current_len)

        return result


@dataclass
class SFTCollator:
    """Collator for SFT that pads input_ids and prompt_mask together.

    Preserves the alignment between input_ids and prompt_mask after padding.

    Args:
        pad_token_id: Token ID for padding.
    """

    pad_token_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        sequences = [item["input_ids"] for item in batch]
        prompt_masks = [item["prompt_mask"] for item in batch]

        input_ids, attention_mask = _pad_sequence(sequences, self.pad_token_id)
        max_len = input_ids.shape[1]
        B = len(sequences)

        # Pad prompt_mask with False (padding positions are not prompt)
        padded_prompt_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, pm in enumerate(prompt_masks):
            padded_prompt_mask[i, : len(pm)] = torch.tensor(pm, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_mask": padded_prompt_mask,
        }


@dataclass
class DPOCollator:
    """Collator for DPO/GRPO that pads chosen and rejected sequences.

    Args:
        pad_token_id: Token ID for padding.
    """

    pad_token_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        chosen = [item["chosen_input_ids"] for item in batch]
        rejected = [item["rejected_input_ids"] for item in batch]
        prompt_masks = [item.get("prompt_mask") for item in batch]

        chosen_ids, chosen_mask = _pad_sequence(chosen, self.pad_token_id)
        rejected_ids, rejected_mask = _pad_sequence(rejected, self.pad_token_id)

        result: dict[str, Tensor] = {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }

        if any(pm is not None for pm in prompt_masks):
            max_len = max(chosen_ids.shape[1], rejected_ids.shape[1])
            B = len(batch)
            padded_pm = torch.zeros(B, max_len, dtype=torch.bool)
            for i, pm in enumerate(prompt_masks):
                if pm is not None:
                    padded_pm[i, : len(pm)] = torch.tensor(pm, dtype=torch.bool)
            result["prompt_mask"] = padded_pm

        return result
