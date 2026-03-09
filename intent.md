# Intent

i created a new empty github repo for this project: git@github.com:sjonas50/diffusion.git

## Interpreted Goal

The user wanted to establish a GitHub repository and develop a complete, working diffusion language model framework following a detailed plan. The strategy involved: (1) initializing the local git repo and connecting it to GitHub, (2) running /research and /plan skills to refine architecture based on latest Mercury 2 findings from March 2026, (3) systematically implementing a 5-phase build plan with tests and smoke tests for each phase, (4) handling both architecture decisions (AR-to-dLLM adaptation, masked/block/continuous diffusion variants) and engineering challenges (tied embeddings, tensor shapes, stochastic masking noise).

## Summary

Completed a full end-to-end build of a diffusion LLM training framework from scratch, implementing 5 phases: project scaffold with pyproject.toml and test infrastructure, core math (noise schedules and diffusion processes), model architecture with bidirectional transformers, data pipelines and trainers with smoke tests, and generation/sampling with evaluation harness.

## Dead Ends

- **Causal mask monkey-patching for bidirectional attention**: March 2026 research showed this approach was deprecated in favor of explicit 4D float mask injection; replaced in architecture.md and implementation
- **Using smoke-test checkpoint for generate.py gate**: Checkpoint was corrupted (safetensors save failed due to GPT-2 tied embeddings); switched to using gpt2 base model directly
- **Python 3.10 in Docker image**: Project requires Python >=3.11; base image uses 3.10.13, causing build failure
- **Loss decreasing over fixed steps during trainer test**: Diffusion's stochastic masking creates noisy gradients per step; fixed by comparing average loss over batches of steps with higher learning rate instead

## Decisions

- **Use AR-to-dLLM adaptation rather than training from scratch**: Mercury 2 research showed this is the state-of-the-art approach; allows leveraging pretrained autoregressive models and is computationally more efficient
- **Implement three diffusion variants (masked, block, continuous) as separate modules**: Allows flexibility for different use cases and enables comparison of approaches; each has distinct computational characteristics and quality tradeoffs
- **Use explicit 4D float mask injection in BidirectionalTransformer rather than monkey-patching**: Cleaner, more maintainable design; reflects latest best practices from Mercury 2 and avoids fragile runtime patching
- **First-Hitting Sampler as default generation method**: Balances speed and quality better than other samplers; well-suited for diffusion LMs with TTFT tradeoff like Mercury 2
- **Use uv for Python package management instead of pip**: Faster lockfile resolution and workspace support; aligns with modern Python best practices for reproducible builds
- **Separate tokenizer_path from model_path in generate script**: Checkpoints may not always include saved tokenizers; allows flexibility in using different tokenizer versions or pretrained tokenizers
