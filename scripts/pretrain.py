"""Pretraining entry point for diffusion LMs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from loguru import logger
from transformers import HfArgumentParser

from diffusion_lm.config.model import ModelConfig
from diffusion_lm.config.training import DiffusionTrainingArguments
from diffusion_lm.trainers.base import DiffusionTrainer


@dataclass
class PretrainScriptArgs:
    """Extra CLI args for the pretrain script (not in TrainingArguments)."""

    model_name_or_path: str = field(default="gpt2", metadata={"help": "Base model."})
    process_type: str = field(default="masked", metadata={"help": "masked | block | continuous"})
    schedule_type: str = field(
        default="linear", metadata={"help": "linear | cosine | loglinear"}
    )
    block_size: int = field(default=512, metadata={"help": "Token block size for pretraining."})
    dtype: str = field(
        default="bfloat16", metadata={"help": "Model weight dtype: float32 | bfloat16 | float16."}
    )
    # Dataset args
    dataset_name: str = field(
        default="", metadata={"help": "HuggingFace dataset name (e.g. ashraq/financial-news-articles)."}  # noqa: E501
    )
    dataset_config: str = field(default="", metadata={"help": "Dataset config/subset."})
    dataset_split: str = field(default="train", metadata={"help": "Dataset split."})
    text_column: str = field(default="text", metadata={"help": "Column containing text."})
    max_train_samples: int = field(
        default=0, metadata={"help": "Cap training samples (0 = use all)."}
    )
    streaming: bool = field(default=False, metadata={"help": "Use streaming dataset."})


def _build_dataset(args: PretrainScriptArgs, tokenizer):
    """Load a HuggingFace dataset and tokenize into fixed-length blocks."""
    import torch
    from torch.utils.data import Dataset

    from diffusion_lm.data.pretraining import tokenize_and_group

    if not args.dataset_name:
        # Fallback: small synthetic dataset for smoke testing
        logger.warning("No --dataset_name provided; using synthetic random data.")
        vocab_size = tokenizer.vocab_size

        class _Synthetic(Dataset):
            def __init__(self):
                import torch
                self.data = torch.randint(1, vocab_size - 1, (256, args.block_size))
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return {"input_ids": self.data[i]}

        return _Synthetic()

    from datasets import load_dataset

    logger.info(f"Loading dataset: {args.dataset_name} split={args.dataset_split}")
    ds_kwargs = {"streaming": args.streaming}
    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config,
                          split=args.dataset_split, **ds_kwargs)
    else:
        ds = load_dataset(args.dataset_name, split=args.dataset_split, **ds_kwargs)

    if args.max_train_samples > 0 and not args.streaming:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
        logger.info(f"Capped to {len(ds)} samples")

    logger.info("Tokenizing and grouping into blocks...")
    chunks = list(tokenize_and_group(
        ds, tokenizer, block_size=args.block_size, text_column=args.text_column
    ))
    logger.info(f"  {len(chunks):,} blocks of {args.block_size} tokens")

    class _ChunkDataset(Dataset):
        def __init__(self, chunks):
            self.chunks = [torch.tensor(c["input_ids"]) for c in chunks]
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i): return {"input_ids": self.chunks[i]}

    return _ChunkDataset(chunks)


def main() -> None:
    parser = HfArgumentParser((PretrainScriptArgs, DiffusionTrainingArguments))

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    script_args, training_args = parser.parse_args_into_dataclasses()

    from transformers import AutoTokenizer

    from diffusion_lm.config.diffusion import DiffusionConfig
    from diffusion_lm.data.collators import RandomTruncateCollator
    from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token
    from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM

    model_config = ModelConfig(
        model_name_or_path=script_args.model_name_or_path,
        init_from_pretrained=True,
        dtype=script_args.dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    backbone = BidirectionalTransformer(model_config)
    mask_token_id = add_mask_token(backbone, tokenizer)

    diffusion_config = DiffusionConfig(
        process_type=script_args.process_type,
        schedule_type=script_args.schedule_type,
        mask_token_id=mask_token_id,
    )

    model = MaskedDiffusionLM(model_config, diffusion_config)
    model.backbone = backbone

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: {script_args.model_name_or_path} | {n_params:,} params"
        f" | mask_token_id={mask_token_id}"
    )

    train_dataset = _build_dataset(script_args, tokenizer)

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RandomTruncateCollator(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()


if __name__ == "__main__":
    main()
