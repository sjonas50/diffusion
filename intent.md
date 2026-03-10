# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

The user wanted to establish a reliable GPU training pipeline for diffusion language models. The strategy was: (1) review and fix the training code for A100 compatibility (FA2 bidirectional attention), (2) deploy to RunPod with proper environment setup, (3) validate the training run completes successfully with good loss convergence, (4) document the entire process for reproducibility, and (5) evaluate the trained model to understand effectiveness.

## Summary

Set up and completed a 2000-step pretraining run of Qwen3-0.6B on an A100 GPU via RunPod, achieving a final loss of 7.52. Created comprehensive documentation for RunPod deployment and Flash Attention 2 setup, then evaluated the trained model through text generation, perplexity measurement, and base-vs-fine-tuned comparison.

## Dead Ends

- **Using `runpod/pytorch:latest` Docker image tag**: Tag doesn't exist; needed explicit version like `2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Using `uv venv` on RunPod to isolate Python environment**: Created duplicate PyTorch installation with wrong CUDA version (cu128 instead of cu124), broke system PyTorch which was already correctly configured
- **Building flash-attn from source via pip**: CUDA kernel compilation took 20-40 minutes; switched to installing prebuilt wheels directly
- **Using 4D attention masks with Flash Attention 2**: FA2 doesn't support masking through attention mask tensor shape; requires explicit `is_causal=False` set on attention modules instead
- **Using evaluate.py for ELBO perplexity without modifications**: Script assumes architecture config.json exists in checkpoint directory, which it doesn't; needed custom config construction like generate.py

## Decisions

- **Set `is_causal=False` on all attention modules during initialization when FA2 is detected, rather than relying on mask tensor shape**: FA2 explicitly requires `is_causal` parameter and ignores attention masks; other backends (eager, sdpa) work with 4D mask tensors, so dual-path implementation was needed
- **Added `--attn_implementation` CLI flag to pretrain.py**: Allows runtime selection between eager (debugging), sdpa (fallback), and flash_attention_2 (performance); critical for A100 optimization
- **Increased block size from 128 to 512 tokens for A100 training**: MPS conservative defaults don't apply to A100; larger blocks improve GPU utilization and throughput without memory issues on 80GB VRAM
- **Used LR=3e-5 with cosine decay from training-recipes.md instead of previous 1e-4**: Docs recommend 3e-5 for 600M AR-to-dLLM adaptation; resulted in smoother convergence (loss 7.52 vs 9.30 in previous run) and better stability
- **Set up system Python directly instead of venv on RunPod**: RunPod PyTorch images come with correctly configured PyTorch+CUDA+cuDNN; virtual environments isolated from these optimizations and caused version conflicts
- **Created `docs/runpod-setup.md` and updated CLAUDE.md with FA2/RunPod learnings**: Documented gotchas (Docker tags, uv venv conflicts, flash-attn build time, HF Trainer logging) so future training runs avoid same pitfalls
