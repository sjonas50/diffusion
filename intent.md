# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

The user wanted to review previous training settings, fix FA2 (Flash Attention 2) support for A100, create a RunPod pod with proper environment setup, execute the training run with proper logging/checkpoint saving, and then comprehensively evaluate the trained model to assess effectiveness.

## Summary

Set up and executed a full A100 GPU training run on RunPod for a diffusion language model on financial data. Successfully trained for 2000 steps (36 minutes, $0.90 cost) with smooth loss convergence (31.72→7.52), then evaluated the model with text generation and perplexity tests.

## Dead Ends

- **Using `uv venv` with `uv pip install` for flash-attn compilation on RunPod**: Created isolated venv with wrong PyTorch version (2.10.0+cu128 instead of system 2.4.1+cu124), and flash-attn source build took 20+ minutes with no completion. Abandoned for system Python with pre-installed PyTorch.
- **Using incorrect RunPod Docker image tag `runpod/pytorch:latest`**: Tag doesn't exist; image pull failed. Switched to verified tag `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **Relying on 4D bidirectional attention mask for Flash Attention 2**: HuggingFace FA2 implementation requires explicit `is_causal=False` on attention modules. The 4D mask approach works for eager/sdpa but not FA2. Fixed by setting `is_causal=False` during model init when FA2 is detected.
- **Debugging missing loss values in training logs**: HF Trainer's tqdm progress bar overwrites console output with carriage returns. Loss metrics are available in `trainer_state.json` within checkpoints, not in raw log file.
- **Using evaluate.py script as-is for perplexity evaluation**: Script tried to load model architecture from checkpoint directory (no config.json), causing 'model_type' lookup failure. Would need architectural fixes similar to generate.py.

## Decisions

- **Reduced learning rate from 1e-4 to 3e-5 for A100 training run**: docs/training-recipes.md recommended 3e-5 for 600M parameter AR-to-dLLM adaptation. Previous MPS run with 1e-4 achieved loss of 9.30 at step 500; new run reached 8.07 at same step, indicating better convergence.
- **Increased block size from 128 to 512 tokens for A100**: 128 was MPS-conservative limit. A100 can handle longer sequences efficiently; 512 improves efficiency by reducing dataset size from 375k→56k blocks while training on same 192M tokens.
- **Added --attn_implementation CLI flag to pretrain.py**: Allows users to switch between eager, sdpa, and flash_attention_2 implementations. Critical for GPU compatibility and performance tuning.
- **Set is_causal=False explicitly on all attention modules when FA2 is detected**: HuggingFace FA2 requires explicit causal masking flags; cannot rely on mask tensor shapes. Ensures bidirectional attention works correctly with FA2 kernels.
- **Used system Python directly instead of uv venv on RunPod**: RunPod's PyTorch image has PyTorch+CUDA pre-configured. Creating isolated uv venv caused version mismatches and extended setup time. System Python avoided environment isolation overhead.
- **Documented all findings in CLAUDE.md, README.md, and new docs/runpod-setup.md**: Captured runpod gotchas (Docker image tags, uv venv issues, flash-attn build times), FA2 bidirectional attention setup, validated training configs, and cost optimization strategies for future runs.
