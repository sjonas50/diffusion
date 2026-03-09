# Intent

i created a new empty github repo for this project: git@github.com:sjonas50/diffusion.git

## Interpreted Goal

The strategy was to create a complete training framework for diffusion-based language models that can adapt from pretrained AR models. The user wanted to build on the latest research (Mercury 2, block diffusion) and structure the codebase with proper configuration management, multiple diffusion variants (masked/block/continuous), and a multi-stage training pipeline (pretrain/SFT/DPO/GRPO).

## Summary

The user initialized a GitHub repository and then conducted an extensive multi-phase development session: researching the latest diffusion LLM techniques (Mercury 2, BD3LM, DPO/GRPO), planning the architecture with updated findings, and building the initial project scaffold with configurations, noise schedules, and diffusion process implementations.

## Dead Ends

- **Using causal mask monkey-patching approach for bidirectional attention**: Research showed this was deprecated; replaced with explicit 4D float mask injection into the backbone
- **Training diffusion LM from scratch without AR initialization**: Research indicated AR-to-dLLM adaptation from pretrained checkpoints is the correct approach (Mercury 2 strategy)

## Decisions

- **Used explicit 4D float mask injection instead of monkey-patching for bidirectional attention**: Cleaner architecture that integrates masks directly into the transformer backbone without runtime hacks
- **Implemented three diffusion variants: masked (Mercury/LLaDA), block (BD3LM), and continuous (Diffusion-LM)**: Allows comparative research across different diffusion approaches while maintaining unified interface
- **Used linear and cosine schedules as primary noise schedules, with log-linear as secondary option**: Linear and cosine are well-validated; log-linear (from MDLM) provided an additional experimental option
- **Structured configs into separate YAML files for models, diffusion processes, and training regimes**: Enables easy experimentation and reproducibility across different model scales (small/medium/large/8B)
- **Created separate trainers for each alignment approach (DPO, GRPO, SFT)**: Reflects the distinct computational patterns and memory requirements of each training paradigm (32 forward passes for DPO/GRPO)
