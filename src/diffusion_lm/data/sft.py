"""SFT data pipeline: prompt/response pairs with prompt_mask boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class SFTDataset:
    """Dataset for supervised fine-tuning with prompt-aware masking.

    Produces (input_ids, prompt_mask) pairs where:
        - input_ids: concatenated prompt + response tokens.
        - prompt_mask: True at prompt positions, False at response positions.

    During training, prompt tokens are never masked and excluded from the loss.
    Only response tokens contribute to the diffusion objective.

    Supports two formats:
        Single-turn: {"prompt": str, "response": str}
        Multi-turn:  {"messages": [{"role": "user"|"assistant", "content": str}]}

    Args:
        data: List of examples in either supported format.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum total sequence length. Longer sequences are truncated.
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
        logger.info(f"SFTDataset: {len(self.examples)} examples, max_length={max_length}")

    def _process(self, example: dict) -> dict[str, list]:
        """Convert one example to (input_ids, prompt_mask)."""
        if "messages" in example:
            return self._process_multiturn(example["messages"])
        return self._process_singleturn(example["prompt"], example["response"])

    def _process_singleturn(self, prompt: str, response: str) -> dict[str, list]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), self.max_length)
        prompt_mask = [True] * prompt_len + [False] * (len(input_ids) - prompt_len)

        return {"input_ids": input_ids, "prompt_mask": prompt_mask}

    def _process_multiturn(self, messages: list[dict]) -> dict[str, list]:
        """Multi-turn: user turns = prompt (True), assistant turns = response (False)."""
        input_ids: list[int] = []
        prompt_mask: list[bool] = []

        for msg in messages:
            content_ids = self.tokenizer.encode(msg["content"], add_special_tokens=False)
            is_prompt = msg["role"] == "user"
            input_ids.extend(content_ids)
            prompt_mask.extend([is_prompt] * len(content_ids))

        input_ids = input_ids[: self.max_length]
        prompt_mask = prompt_mask[: self.max_length]
        return {"input_ids": input_ids, "prompt_mask": prompt_mask}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list]:
        return self.examples[idx]
