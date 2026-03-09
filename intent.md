# Intent

i created a new empty github repo for this project: git@github.com:sjonas50/diffusion.git

## Interpreted Goal

Create a production-ready training framework for diffusion-based LLMs (inspired by Mercury 2) that supports AR-to-dLLM adaptation from pretrained checkpoints. Strategy: (1) Research latest Mercury 2/diffusion LLM architectures and correct the plan, (2) Generate architecture/build-plan docs, (3) Scaffold project with configs and test framework, (4) Implement core math (schedules, diffusion processes), (5) Build bidirectional transformer with explicit 4D attention masks, (6) Implement data pipeline and trainers, (7) Create samplers for generation. All phases tested end-to-end.

## Summary

Built a complete diffusion language model training framework from scratch, including configuration system, noise schedules, three diffusion variants (masked, block, continuous), bidirectional transformer backbone, five trainer types (pretraining, SFT, DPO, GRPO), five sampling strategies, and end-to-end data pipeline with working pretrain and generate scripts.

## Dead Ends

- **Using monkey-patched `_update_causal_mask = nop` to disable causal masking in transformer**: Research revealed this approach is deprecated and fragile; explicit 4D float mask injection is the modern standard used in Mercury 2
- **Training on fixed batch expecting monotonic loss decrease in diffusion LM**: Diffusion's stochastic timestep sampling and random masking patterns cause noisy per-step loss; switched to comparing 5-step averages with higher learning rate
- **Using GPT-2's tied weights (wte == lm_head) with safetensors checkpoint saving**: safetensors can't serialize tied weights; switched to loading/saving via transformers library which handles weight sharing correctly

## Decisions

- **Used explicit 4D float mask injection instead of monkey-patching for bidirectional attention**: Based on research findings that Mercury 2 and modern diffusion LLMs inject float masks directly into attention computations for clarity and compatibility
- **Implemented three diffusion variants (masked, block BD3LM, continuous) as separate model classes**: Allows users to experiment with different diffusion processes; each has different mathematical properties and sampling speeds suitable for different use cases
- **Built AR-to-dLLM adaptation pattern where pretrained AR checkpoints are loaded and converted**: Mercury 2 research showed adaptation from pretrained LLaMA-scale models is more practical and faster than training diffusion LMs from scratch
- **Implemented First-Hitting Sampler as default with explicit mask token exclusion from predictions**: Most sample-efficient strategy for masked diffusion; excluding mask_token_id from logits prevents model from predicting mask tokens at inference
- **Separated trainer classes (PretrainingTrainer, SFTTrainer, DPOTrainer, GRPOTrainer) instead of single unified trainer**: Each training phase has different data requirements, loss computation, and masking patterns; separation allows cleaner, more maintainable implementations
