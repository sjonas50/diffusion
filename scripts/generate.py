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

    # Use tokenizer_path as the architecture source (base model).
    # model_path may be a local checkpoint dir without config.json — in that case
    # we build the architecture from tok_path, add [MASK], then load the checkpoint weights.
    arch_path = tok_path  # always use the base model for architecture
    model_config = ModelConfig(model_name_or_path=arch_path)

    # Build backbone, add mask token (resizes embeddings), build full model
    backbone = BidirectionalTransformer(model_config)
    if tokenizer.mask_token_id is None:
        mask_token_id = add_mask_token(backbone, tokenizer)
    else:
        mask_token_id = tokenizer.mask_token_id

    diffusion_config = DiffusionConfig(mask_token_id=mask_token_id)
    model = MaskedDiffusionLM(model_config, diffusion_config)
    model.backbone = backbone

    # Load fine-tuned weights from checkpoint if model_path differs from base model
    if args.model_path != arch_path:
        import os
        ckpt_file = os.path.join(args.model_path, "model.safetensors")
        if not os.path.exists(ckpt_file):
            # Try pytorch_model.bin fallback
            ckpt_file = os.path.join(args.model_path, "pytorch_model.bin")
        if os.path.exists(ckpt_file):
            from loguru import logger
            logger.info(f"Loading checkpoint weights from {ckpt_file}")
            if ckpt_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_file, device=str(device))
            else:
                state_dict = torch.load(ckpt_file, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                n = len(unexpected)
                logger.warning(f"Unexpected keys ({n}): {unexpected[:5]}{'...' if n > 5 else ''}")
        else:
            from loguru import logger
            logger.warning(f"No checkpoint weights found in {args.model_path}, using base weights")

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
