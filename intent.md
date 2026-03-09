# Intent

i created a new empty github repo for this project: git@github.com:sjonas50/diffusion.git

## Interpreted Goal

Set up a professional Python ML project from scratch with: (1) GitHub integration, (2) comprehensive research on Mercury 2 and diffusion LLMs as of March 2026, (3) architecture and build planning informed by that research, (4) phased implementation starting with configs/schedules/diffusion processes, then model components, with full test coverage and linting enforcement.

## Summary

Initialized a GitHub-connected local repository and built a complete foundation for a Diffusion LLM training framework, including research synthesis, architecture design, and three phases of implementation (project scaffold, core math, and model architecture) with passing tests.

## Dead Ends

- **Using `_update_causal_mask = nop` monkey-patch for disabling causal attention in bidirectional transformer**: Research found this approach is deprecated; replaced with explicit 4D float mask injection into attention computation instead
- **Training diffusion LMs from scratch with random initialization**: Research determined AR-to-dLLM adaptation (from pretrained AR checkpoints) is the recommended path; architectural plan shifted to this approach
- **Using masked sampler as only generation method**: Research findings and architecture updates introduced First-Hitting Sampler and block diffusion samplers as alternatives; masked sampler is now one of several options

## Decisions

- **Adopted three diffusion variants (masked, block/BD3LM, continuous) with factory pattern registry**: Research showed each variant has different speed/quality tradeoffs; providing all three allows experimentation and comparison on the same codebase
- **Implemented BidirectionalTransformer with explicit 4D float mask injection rather than attention mechanism patching**: Cleaner, more maintainable, and aligns with latest research; mask is injected before softmax in the attention computation
- **Used AR-to-dLLM adaptation pattern (loading pretrained checkpoints) rather than training from scratch**: Research found Mercury 2 and other recent diffusion LLMs use this transfer learning approach; more practical for resource-constrained environments
- **Pinned transformers<5.10 to avoid breaking changes in HuggingFace API**: Research identified recent versions have major API shifts; locking version ensures reproducibility and stability
- **Created five trainer classes (pretrain, SFT, DPO, GRPO, base) with separate configs**: Each training phase has different objectives and hyperparameter requirements; separate trainers allow clean abstraction and config management
- **Used pytest module-scoped fixtures for heavy objects (backbone model) and function-scoped for test data**: Balances test isolation with execution speed; model initialization is expensive, but test data should vary per test
