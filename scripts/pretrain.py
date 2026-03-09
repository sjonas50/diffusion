"""Pretraining entry point for diffusion LMs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from diffusion_lm.config.diffusion import DiffusionConfig
from diffusion_lm.config.model import ModelConfig
from diffusion_lm.config.training import DiffusionTrainingArguments
from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM
from diffusion_lm.trainers.base import DiffusionTrainer


@dataclass
class PretrainScriptArgs:
    """Extra CLI args for the pretrain script (not in TrainingArguments)."""

    model_name_or_path: str = field(default="gpt2", metadata={"help": "Base model."})
    process_type: str = field(default="masked", metadata={"help": "masked | block | continuous"})
    schedule_type: str = field(
        default="linear", metadata={"help": "linear | cosine | loglinear"}
    )
    block_size: int = field(default=2048, metadata={"help": "Token block size for pretraining."})


def main() -> None:
    parser = HfArgumentParser((PretrainScriptArgs, DiffusionTrainingArguments))

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    script_args, training_args = parser.parse_args_into_dataclasses()

    # Build configs
    model_config = ModelConfig(
        model_name_or_path=script_args.model_name_or_path,
        init_from_pretrained=True,
    )

    from diffusion_lm.config.diffusion import DiffusionConfig

    diffusion_config = DiffusionConfig(
        process_type=script_args.process_type,
        schedule_type=script_args.schedule_type,
    )

    # Build model
    if script_args.process_type == "masked":
        from transformers import AutoTokenizer

        from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token
        from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM

        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

        # Build backbone first to resize its embeddings, then get mask_token_id
        backbone = BidirectionalTransformer(model_config)
        mask_token_id = add_mask_token(backbone, tokenizer)
        diffusion_config.mask_token_id = mask_token_id

        model = MaskedDiffusionLM(model_config, diffusion_config)
        # Replace the auto-constructed backbone with the already-resized one
        model.backbone = backbone
    else:
        raise NotImplementedError(f"process_type={script_args.process_type} not yet supported")

    # Build a small synthetic dataset for smoke-testing (no real data needed)
    import torch
    from torch.utils.data import Dataset

    class _SyntheticDataset(Dataset):
        def __init__(self, vocab_size: int, seq_len: int = 128, n: int = 64) -> None:
            self.data = torch.randint(1, vocab_size - 1, (n, seq_len))

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict:
            return {"input_ids": self.data[idx]}

    vocab_size = model.backbone.transformer.config.vocab_size
    train_dataset = _SyntheticDataset(vocab_size)

    from diffusion_lm.data.collators import RandomTruncateCollator

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RandomTruncateCollator(pad_token_id=0),
    )

    trainer.train()


if __name__ == "__main__":
    main()
