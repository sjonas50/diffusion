# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

Review previous training setup, fix Flash Attention 2 compatibility for A100, deploy to RunPod with proper environment setup, run a complete training job, monitor convergence, and evaluate the trained model's effectiveness on financial text generation.

## Summary

Completed a full A100 GPU training run of a diffusion LM on financial news data (2000 steps, 36 minutes), achieving smooth convergence (loss 31.72→7.52), then evaluated the model through text generation and compared base vs fine-tuned outputs.

## Dead Ends

- **Using `uv venv` on RunPod for isolated Python environment**: Created wrong PyTorch version (2.10.0+cu128 instead of 2.4.1+cu124), no pip binary, and flash-attn source build took 20+ minutes without completing
- **Building flash-attn from source via `uv pip install --no-build-isolation`**: CUDA kernel compilation took 20-40 minutes; switching to prebuilt wheels from PyPI was much faster
- **Initial RunPod pod creation with incorrect Docker image tags**: Tags like `runpod/pytorch:latest` don't exist; had to use validated tag `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Relying on 4D mask approach for Flash Attention 2 bidirectional attention**: FA2 requires explicit `is_causal=False` parameter on attention layers; 4D masks work for eager/SDPA but not FA2
- **Running `evaluate.py` as-is on checkpoint**: Script tries to load model architecture from checkpoint dir which lacks `config.json`; needs `--model_name_or_path` to specify architecture source

## Decisions

- **Set `is_causal=False` on all attention modules in `BidirectionalTransformer` when using Flash Attention 2**: FA2 kernel doesn't support mask-based bidirectional attention like eager mode; must use explicit `is_causal=False` parameter, matching LLaDA/Dream implementation
- **Use system Python directly on RunPod instead of creating isolated venv**: RunPod PyTorch image comes with correctly configured PyTorch+CUDA; isolated venvs don't preserve these system optimizations and add setup complexity
- **Added `--attn_implementation` CLI flag to pretrain.py**: Allows easy switching between eager, SDPA, and flash_attention_2 implementations without code changes; essential for A100 where FA2 provides significant speedup
- **Lowered learning rate from 1e-4 to 3e-5 per training-recipes.md guidance**: Documentation recommends 3e-5 for 600M AR-to-dLLM adaptation; empirically produces better convergence (loss 7.52 vs previous 9.30 at step 500)
- **Increased block_size from 128 to 512 tokens for A100**: MPS-conservative 128 tokens underutilizes A100 memory; 512 tokens better matches A100 capabilities and improves throughput (verified 29.4 samples/sec)
- **Documented entire RunPod setup process including gotchas in docs/runpod-setup.md**: Future training runs need this institutional knowledge; prevents repeating setup mistakes (Docker image tags, uv vs system Python, flash-attn source builds)
