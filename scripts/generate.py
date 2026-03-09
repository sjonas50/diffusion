"""Generation entry point for diffusion LMs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, HfArgumentParser

from diffusion_lm.config.generation import GenerationConfig


@dataclass
class GenerateArgs:
    """CLI arguments for generate.py."""

    model_path: str = field(metadata={"help": "Path to trained MaskedDiffusionLM checkpoint."})
    tokenizer_path: str = field(
        default="", metadata={"help": "Tokenizer path (defaults to model_path)."}
    )
    prompt: str = field(default="Hello", metadata={"help": "Prompt text."})
    max_new_tokens: int = field(default=64, metadata={"help": "Tokens to generate."})
    num_steps: int = field(default=32, metadata={"help": "Denoising steps."})
    sampler: str = field(
        default="first_hitting",
        metadata={"help": "Sampler: first_hitting | block | continuous | cached"},
    )
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    guidance_scale: float = field(default=0.0, metadata={"help": "Classifier-free guidance scale."})
    running_confidence_remasking: bool = field(
        default=True, metadata={"help": "Enable running confidence remasking."}
    )
    device: str = field(default="cpu", metadata={"help": "Device: cpu | cuda | mps."})


def build_sampler(sampler_name: str):
    """Instantiate the requested sampler."""
    if sampler_name == "first_hitting":
        from diffusion_lm.samplers.first_hitting_sampler import FirstHittingSampler

        return FirstHittingSampler()
    if sampler_name == "block":
        from diffusion_lm.samplers.block_sampler import BlockSampler

        return BlockSampler()
    if sampler_name == "continuous":
        from diffusion_lm.samplers.continuous_sampler import ContinuousSampler

        return ContinuousSampler()
    if sampler_name == "cached":
        from diffusion_lm.samplers.cached_sampler import CachedSampler

        return CachedSampler()
    raise ValueError(f"Unknown sampler: {sampler_name!r}")


def main() -> None:
    parser = HfArgumentParser(GenerateArgs)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    (args,) = parser.parse_args_into_dataclasses()
    device = torch.device(args.device)

    # Load tokenizer and model from checkpoint
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    from diffusion_lm.config.diffusion import DiffusionConfig
    from diffusion_lm.config.model import ModelConfig
    from diffusion_lm.models.backbone import BidirectionalTransformer, add_mask_token
    from diffusion_lm.models.masked_diffusion_lm import MaskedDiffusionLM

    model_config = ModelConfig(model_name_or_path=args.model_path)

    # Build backbone first; add mask token (resizes embeddings); then build full model
    backbone = BidirectionalTransformer(model_config)
    if tokenizer.mask_token_id is None:
        mask_token_id = add_mask_token(backbone, tokenizer)
    else:
        mask_token_id = tokenizer.mask_token_id

    diffusion_config = DiffusionConfig(mask_token_id=mask_token_id)
    model = MaskedDiffusionLM(model_config, diffusion_config)
    # Replace the auto-constructed backbone with the already-resized one
    model.backbone = backbone
    model.eval()
    model.to(device)

    # Tokenize prompt
    enc = tokenizer(args.prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"].to(device)

    # Build generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_steps=args.num_steps,
        temperature=args.temperature,
        sampler=args.sampler,
        running_confidence_remasking=args.running_confidence_remasking,
        guidance_scale=args.guidance_scale,
    )

    sampler = build_sampler(args.sampler)

    with torch.no_grad():
        output = sampler.generate(model, prompt_ids, gen_config)

    sequences = output.sequences
    for i, seq in enumerate(sequences):
        decoded = tokenizer.decode(seq, skip_special_tokens=False)
        print(f"[{i}] {decoded}")


if __name__ == "__main__":
    main()
