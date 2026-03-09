"""Pretraining data pipeline: streaming tokenize-and-group."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from datasets import IterableDataset
    from transformers import PreTrainedTokenizerBase


def tokenize_and_group(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 2048,
    text_column: str = "text",
) -> Iterator[dict[str, list[int]]]:
    """Tokenize a streaming dataset and group tokens into fixed-length blocks.

    Concatenates all text, tokenizes, and yields non-overlapping chunks of
    exactly `block_size` tokens. The final incomplete chunk is discarded.

    This is the standard pretraining data preparation used by LLaDA and GPT-2.

    Args:
        dataset: HuggingFace IterableDataset with a text column.
        tokenizer: Tokenizer to use for encoding.
        block_size: Number of tokens per chunk.
        text_column: Name of the text field in the dataset.

    Yields:
        Dicts of {"input_ids": List[int]} with exactly block_size tokens.
    """
    buffer: list[int] = []

    for example in dataset:
        text = example[text_column]
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)

        while len(buffer) >= block_size:
            yield {"input_ids": buffer[:block_size]}
            buffer = buffer[block_size:]

    logger.debug(f"tokenize_and_group: discarded {len(buffer)} trailing tokens")


class PretrainingDataset:
    """Streaming pretraining dataset that yields fixed-length token chunks.

    Args:
        dataset_name: HuggingFace dataset name (e.g. "wikitext").
        dataset_config: Dataset configuration (e.g. "wikitext-2-raw-v1").
        tokenizer: Tokenizer for encoding.
        block_size: Tokens per chunk (default 2048).
        split: Dataset split to use (default "train").
        streaming: If True, use streaming mode (default True).
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str | None,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 2048,
        split: str = "train",
        streaming: bool = True,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Install datasets: uv add datasets") from e

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=False,
        )
        logger.info(
            f"PretrainingDataset: {dataset_name}/{dataset_config}, "
            f"split={split}, block_size={block_size}, streaming={streaming}"
        )

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        yield from tokenize_and_group(self.dataset, self.tokenizer, self.block_size)
