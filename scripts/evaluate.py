"""Evaluation entry point for diffusion LMs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer, HfArgumentParser

from diffusion_lm.evaluation.perplexity import ELBOPerplexity


@dataclass
class EvaluateArgs:
    """CLI arguments for evaluate.py."""

    model_path: str = field(metadata={"help": "Path to trained MaskedDiffusionLM or HF model ID."})
    tokenizer_path: str = field(
        default="", metadata={"help": "Tokenizer path (defaults to model_path)."}
    )
    benchmarks: str = field(
        default="perplexity",
        metadata={"help": "Comma-separated benchmarks: perplexity,lm_eval"},
    )
    dataset: str = field(default="wikitext", metadata={"help": "HuggingFace dataset name."})
    dataset_config: str = field(
        default="wikitext-2-raw-v1", metadata={"help": "Dataset config/subset."}
    )
    split: str = field(default="test", metadata={"help": "Dataset split to evaluate on."})
    batch_size: int = field(default=4, metadata={"help": "Evaluation batch size."})
    num_timestep_samples: int = field(
        default=8, metadata={"help": "MC timestep samples for ELBO estimation."}
    )
    output_file: str = field(
        default="", metadata={"help": "JSON file to write results (defaults to stdout)."}
    )
    device: str = field(default="cpu", metadata={"help": "Compute device."})
    max_eval_samples: int = field(
        default=100, metadata={"help": "Max samples to evaluate (0=all)."}
    )


def load_model(args: EvaluateArgs, tokenizer):
    """Load model with mask token setup."""
    from diffusion_lm.config.diffusion import DiffusionConfig
    from diffusion_lm.config.model import ModelConfig
    from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token
    from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM

    model_config = ModelConfig(model_name_or_path=args.model_path)
    backbone = BidirectionalTransformer(model_config)

    if tokenizer.mask_token_id is None:
        mask_token_id = add_mask_token(backbone, tokenizer)
    else:
        mask_token_id = tokenizer.mask_token_id

    diffusion_config = DiffusionConfig(mask_token_id=mask_token_id)
    model = MaskedDiffusionLM(model_config, diffusion_config)
    model.backbone = backbone
    return model


def build_eval_dataset(args: EvaluateArgs, tokenizer):
    """Build a tokenized evaluation dataset."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("Install datasets: uv add datasets") from err

    from torch.utils.data import Dataset

    raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=False)
    if args.max_eval_samples > 0:
        raw = raw.select(range(min(args.max_eval_samples, len(raw))))

    block_size = 128

    class _TokenizedDataset(Dataset):
        def __init__(self, examples):
            self.input_ids = []
            buffer = []
            for ex in examples:
                text = ex.get("text", "")
                if not text:
                    continue
                toks = tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(toks)
                while len(buffer) >= block_size:
                    self.input_ids.append(torch.tensor(buffer[:block_size]))
                    buffer = buffer[block_size:]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx]}

    return _TokenizedDataset(raw)


def main() -> None:
    parser = HfArgumentParser(EvaluateArgs)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    (args,) = parser.parse_args_into_dataclasses()

    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    model = load_model(args, tokenizer)
    model.eval()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    results: dict = {}

    if "perplexity" in benchmarks:
        logger.info(
            f"Running ELBO perplexity on {args.dataset}/{args.dataset_config} ({args.split})"
        )
        dataset = build_eval_dataset(args, tokenizer)

        if len(dataset) == 0:
            logger.warning("Dataset is empty — skipping perplexity")
            results["perplexity"] = {"ppl_bound": None, "nll_per_token": None}
        else:
            evaluator = ELBOPerplexity(
                num_timestep_samples=args.num_timestep_samples,
                batch_size=args.batch_size,
            )
            ppl_results = evaluator.compute(model, dataset, device=args.device)
            results["perplexity"] = ppl_results
            logger.info(f"PPL bound: {ppl_results['ppl_bound']:.2f}")

    if "lm_eval" in benchmarks:
        logger.warning("lm_eval benchmark requires 'pip install lm-eval'; skipping for now.")
        results["lm_eval"] = {"status": "not_installed"}

    # Output results
    output_json = json.dumps(results, indent=2)
    if args.output_file:
        Path(args.output_file).write_text(output_json)
        logger.info(f"Results written to {args.output_file}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
