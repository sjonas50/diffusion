# Intent

i created a new empty github repo for this project: git@github.com:sjonas50/diffusion.git

## Interpreted Goal

The user wanted to establish a production-ready diffusion language model training framework. The strategy involved: (1) connecting local code to GitHub, (2) researching current state-of-art in diffusion LLMs to validate and update the existing plan, (3) designing the architecture based on AR-to-dLLM adaptation rather than training from scratch, (4) scaffolding the entire project structure with proper Python tooling (pyproject.toml, ruff, pytest), configuration templates for four diffusion variants and five training modes, and entry point scripts.

## Summary

Initialized a diffusion LLM training framework project by setting up GitHub repository, researching latest Mercury 2 and diffusion LLM approaches (March 2026), and scaffolding the complete project structure with dependencies, configs, and test files.

## Dead Ends

- **Using monkey-patched causal mask disabling (`_update_causal_mask = nop`)**: Research revealed this approach was deprecated in favor of explicit 4D float mask injection into the attention mechanism, which is cleaner and more reliable
- **Assuming Mercury 2 achieves GPT-4o-class quality**: Latest research (Feb 2026) showed Mercury 2 achieves AIME 91.1 and GPQA 73.6 (Claude Haiku tier), not GPT-4o tier, with 3.48s TTFT due to all diffusion steps before first token

## Decisions

- **Center architecture on AR-to-dLLM adaptation from pretrained checkpoints rather than training from scratch**: Research findings on Mercury 2 and block diffusion methods showed that adapting existing AR models is more practical and achieves better results than training diffusion LMs from random initialization
- **Support three diffusion process variants (masked, block, continuous) with corresponding samplers**: The research identified multiple viable approaches; offering all three allows experimentation while First-Hitting Sampler serves as the default based on latest best practices
- **Use uv for Python dependency management with pinned transformers<5.10**: Ensures reproducibility across environments and prevents API breakage from newer transformer versions
- **Implement five trainers (pretrain, SFT, DPO, GRPO, base) with separate configuration files**: Modular trainer design allows independent optimization of each alignment stage based on the build plan phases
