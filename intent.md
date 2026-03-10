# Intent

i created a new empty github repo for this project: git@github.com:sjonas50/diffusion.git

## Interpreted Goal

The user wanted to build a production-ready diffusion LLM training framework inspired by Mercury 2 (arXiv 2506+, Feb 2026). The strategy was: (1) research latest diffusion LLM techniques and constraints, (2) design architecture centered on AR-to-dLLM adaptation with bidirectional attention and multiple diffusion variants (masked, block, continuous), (3) implement modular components (configs, schedules, diffusion processes, models, data pipelines, trainers, samplers), (4) validate with end-to-end smoke tests, (5) harden with Docker, evaluation tools, and comprehensive test coverage.

## Summary

Initialized a new diffusion LLM training framework project: connected empty GitHub repo, conducted research on Mercury 2 and diffusion LLMs, and built a complete training infrastructure through 5 development phases (project scaffold, core math, model architecture, data/training pipelines, and generation/sampling). Achieved 54/54 tests passing with working smoke tests for pretraining and generation.

## Dead Ends

- **Using causal mask monkey-patch (_update_causal_mask = nop) for bidirectional attention in transformers**: Research revealed this approach is deprecated; replaced with explicit 4D float mask injection to transformer forward pass per Mercury 2 design
- **Training from scratch vs. AR-to-dLLM adaptation**: Research findings showed Mercury 2 adapts pretrained AR models; updated plan to initialize from GPT-2/LLaMA checkpoints rather than random init
- **Single diffusion process (masked only)**: Research identified block diffusion (BD3LM, arXiv 2503.09573) and continuous diffusion as competitive; expanded architecture to support all three variants
- **Using no_cuda argument in training arguments**: Transformers library removed no_cuda in recent versions; switched to use_cpu parameter
- **Checkpointing weights with tied embeddings (GPT-2 style wte==lm_head)**: safetensors fails on tied weights in tied embedding models; workaround accepted (training still succeeds)

## Decisions

- **Centered architecture on BidirectionalTransformer wrapper that patches CausalLM models with explicit 4D float mask injection in forward pass**: Allows reuse of pretrained AR checkpoints (GPT-2, LLaMA) while enabling bidirectional attention via mask control, avoiding monkey-patching and enabling clean AR-to-dLLM adaptation
- **Implemented three diffusion variants (masked, block/BD3LM, continuous) with pluggable sampler strategies (FirstHitting, Block, Continuous, Cached)**: Research showed Mercury 2 uses masked diffusion but BD3LM shows competitive results; providing all variants lets users benchmark and choose based on their use case (speed vs. quality)
- **Five specialized trainers (Pretraining, SFT, DPO, GRPO, Base) extending HuggingFace Trainer with custom compute_loss overrides**: Diffusion LMs require different loss computation and sampling strategies per phase; modular trainers keep each phase's logic isolated and testable
- **Used antithetic timestep sampling in diffusion (random t per batch) rather than fixed t per step**: Improves gradient variance across timesteps during training; aligns with Mercury 2 and best practices in diffusion model training
- **Pinned transformers<5.10 to avoid API breakage (no_cuda removal, AutoConfig changes)**: Diffusion LM development requires stable transformer internals for mask injection and forward pass control; pinning prevents unexpected failures during dependency updates
- **Deferred Docker Python version fix to allow both 3.10 and 3.11 in pyproject.toml**: Maximizes compatibility with existing CUDA base images (which often ship Python 3.10); can upgrade after NVIDIA releases official Python 3.11+ CUDA images
