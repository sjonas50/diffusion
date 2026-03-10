"""Pretraining entry point for diffusion LMs.

Usage:
    uv run python scripts/pretrain.py \
        --model_name_or_path gpt2 \
        --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
        --max_steps 1000 --output_dir ./checkpoints/pretrain

Resume from checkpoint:
    uv run python scripts/pretrain.py \
        --model_name_or_path gpt2 \
        --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
        --max_steps 2000 --output_dir ./checkpoints/pretrain \
        --resume_from_checkpoint ./checkpoints/pretrain/checkpoint-1000
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from loguru import logger
from transformers import AutoTokenizer, HfArgumentParser

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.config.training import DiffusionTrainingArguments
from diffusion_lm.data.collators import RandomTruncateCollator
from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM
from diffusion_lm.trainers.base import DiffusionTrainer


@dataclass
class PretrainScriptArgs:
    """CLI args for the pretrain script (beyond HF TrainingArguments)."""

    # Model
    model_name_or_path: str = field(
        default="gpt2", metadata={"help": "HF model name or local path."}
    )
    process_type: str = field(default="masked", metadata={"help": "masked | block | continuous"})
    schedule_type: str = field(default="linear", metadata={"help": "linear | cosine | loglinear"})
    dtype: str = field(
        default="float32",
        metadata={"help": "Model dtype: float32 | bfloat16 | float16."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention backend: eager | sdpa | flash_attention_2."},
    )
    block_size: int = field(
        default=512, metadata={"help": "Token block size for pretraining data."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA adapters."})
    lora_rank: int = field(default=16, metadata={"help": "LoRA rank."})

    # Dataset (comma-separated for multi-dataset, e.g. "ds1,ds2")
    dataset_name: str = field(
        default="",
        metadata={"help": "HF dataset name(s), comma-separated."},
    )
    dataset_config: str = field(
        default="",
        metadata={"help": "Dataset config(s), comma-separated. Broadcast last value."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split."})
    text_column: str = field(
        default="text",
        metadata={"help": "Text column name(s), comma-separated. Broadcast last value."},
    )
    max_train_samples: str = field(
        default="0",
        metadata={"help": "Cap training samples per dataset, comma-separated (0 = all)."},
    )
    streaming: bool = field(default=False, metadata={"help": "Stream dataset."})

    # Resume
    ignore_optimizer_on_resume: bool = field(
        default=False,
        metadata={
            "help": "Delete optimizer/scheduler state from checkpoint before resuming. "
            "Useful when switching optimizers (e.g. AdamW → Adafactor)."
        },
    )


def _broadcast(lst: list[str], n: int, default: str) -> list[str]:
    """Pad a list to length n by repeating the last value (or default)."""
    if not lst or lst == [""]:
        return [default] * n
    while len(lst) < n:
        lst.append(lst[-1])
    return lst


def _build_dataset(args: PretrainScriptArgs, tokenizer) -> torch.utils.data.Dataset:
    """Load one or more HF datasets and tokenize into fixed-length blocks.

    Supports comma-separated --dataset_name for multi-dataset training.
    Datasets are tokenized sequentially and concatenated; HF Trainer shuffles
    at the batch level so domains get mixed during training.
    """
    from itertools import chain

    from torch.utils.data import Dataset

    from diffusion_lm.data.pretraining import tokenize_and_group

    if not args.dataset_name:
        logger.warning("No --dataset_name; using synthetic random data for smoke test.")
        vocab_size = tokenizer.vocab_size

        class _Synthetic(Dataset):
            def __init__(self):
                self.data = torch.randint(1, vocab_size - 1, (256, args.block_size))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, i):
                return {"input_ids": self.data[i]}

        return _Synthetic()

    from datasets import load_dataset

    # Parse comma-separated multi-dataset args
    names = [s.strip() for s in args.dataset_name.split(",")]
    configs = _broadcast(
        [s.strip() for s in args.dataset_config.split(",")], len(names), ""
    )
    text_cols = _broadcast(
        [s.strip() for s in args.text_column.split(",")], len(names), "text"
    )
    max_samples_list = _broadcast(
        [s.strip() for s in args.max_train_samples.split(",")], len(names), "0"
    )

    # Build one tokenize_and_group generator per dataset, then chain
    generators = []
    ds_kwargs: dict = {"streaming": args.streaming}

    for name, config, text_col, max_samp_str in zip(
        names, configs, text_cols, max_samples_list
    ):
        max_samp = int(max_samp_str)
        logger.info(
            f"Loading dataset: {name} (config={config or 'none'}, "
            f"text_col={text_col}, max_samples={max_samp or 'all'})"
        )
        if config:
            ds = load_dataset(name, config, split=args.dataset_split, **ds_kwargs)
        else:
            ds = load_dataset(name, split=args.dataset_split, **ds_kwargs)

        if max_samp > 0 and not args.streaming:
            ds = ds.select(range(min(max_samp, len(ds))))
            logger.info(f"  Capped to {len(ds)} samples")

        generators.append(
            tokenize_and_group(
                ds, tokenizer, block_size=args.block_size, text_column=text_col
            )
        )

    logger.info("Tokenizing and grouping into blocks...")
    chunks = list(chain(*generators))
    total_tokens = len(chunks) * args.block_size
    logger.info(
        f"  {len(chunks):,} blocks of {args.block_size} tokens "
        f"({total_tokens / 1e6:.0f}M tokens total)"
    )

    class _ChunkDataset(Dataset):
        def __init__(self, data):
            self.chunks = [torch.tensor(c["input_ids"]) for c in data]

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, i):
            return {"input_ids": self.chunks[i]}

    return _ChunkDataset(chunks)


def _strip_optimizer_state(checkpoint_dir: str) -> None:
    """Remove optimizer/scheduler state from a checkpoint directory.

    This allows resuming training with the model weights intact but a fresh
    optimizer — necessary when switching optimizers (e.g. AdamW → Adafactor).
    """
    for fname in ("optimizer.pt", "scheduler.pt"):
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            backup = path + ".bak"
            os.rename(path, backup)
            logger.info(f"Moved {fname} → {fname}.bak (fresh optimizer on resume)")


def main() -> None:
    parser = HfArgumentParser((PretrainScriptArgs, DiffusionTrainingArguments))

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    script_args, training_args = parser.parse_args_into_dataclasses()

    # Handle optimizer state stripping before HF Trainer tries to load it
    if script_args.ignore_optimizer_on_resume and training_args.resume_from_checkpoint:
        _strip_optimizer_state(training_args.resume_from_checkpoint)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Model (single construction via from_configs) ---
    model_config = ModelConfig(
        model_name_or_path=script_args.model_name_or_path,
        init_from_pretrained=True,
        dtype=script_args.dtype,
        attn_implementation=script_args.attn_implementation,
        use_lora=script_args.use_lora,
        lora_rank=script_args.lora_rank,
    )
    diffusion_config = DiffusionConfig(
        process_type=script_args.process_type,
        schedule_type=script_args.schedule_type,
    )

    model = MaskedDiffusionLM.from_configs(model_config, diffusion_config, tokenizer)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: {script_args.model_name_or_path} | {n_params:,} params | "
        f"mask_token_id={model.diffusion.mask_token_id}"
    )

    # --- Dataset ---
    train_dataset = _build_dataset(script_args, tokenizer)

    # --- Trainer ---
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RandomTruncateCollator(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save final model + tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Saved model and tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    main()
