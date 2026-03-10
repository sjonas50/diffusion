# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

User wanted to review a previous training run, set it up for A100 GPU execution, deploy to RunPod cloud, monitor training progress, and document lessons learned. Strategy involved: (1) auditing existing code for GPU compatibility issues, (2) fixing FA2 bidirectional attention handling, (3) setting up RunPod infrastructure with correct Docker image and dependencies, (4) running full 2000-step training with checkpoint monitoring, (5) performing qualitative and quantitative model evaluations, and (6) documenting the entire process.

## Summary

Successfully deployed and completed a 2000-step training run of a diffusion LM on A100 GPU, achieving final loss of 7.52. Discovered and fixed Flash Attention 2 bidirectional attention setup issues, created comprehensive RunPod deployment documentation, and evaluated the trained model.

## Dead Ends

- **Using `uv venv` on RunPod with system PyTorch**: uv venv creates isolated environment that re-downloads PyTorch 2.10.0 instead of using pre-installed 2.4.1+cu124, breaking flash-attn compatibility
- **Building flash-attn from source on RunPod**: CUDA kernel compilation takes 20-40 minutes; pre-built wheels available from GitHub releases are faster
- **Using wrong Docker image tags (runpod/pytorch:latest, runpod/pytorch:latest-py311-cu12, etc.)**: Non-existent tags caused prolonged initialization; correct tag is runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- **Expecting loss values in stdout logs**: HF Trainer output overwritten by tqdm's carriage returns; loss values only appear in trainer_state.json in checkpoint directories
- **Using checkpoint directory as model architecture source for evaluation**: Checkpoint dirs don't contain config.json with model_type; must use tokenizer_path or original model_name_or_path

## Decisions

- **Use system Python directly instead of uv venv on RunPod**: RunPod PyTorch images have pre-installed, properly configured PyTorch+CUDA; venv creates isolation that breaks compatibility
- **Set `is_causal=False` on attention layers when FA2 is enabled**: 4D mask approach works for eager/sdpa but FA2 requires explicit causal=False parameter; aligns with LLaDA/Dream approach
- **Add `--attn_implementation` CLI flag to pretrain.py**: Enables runtime switching between eager/sdpa/flash_attention_2 without code changes; essential for different hardware
- **Fix padding mask value from -1e9 to torch.finfo(dtype).min**: Prevents numerical issues with bfloat16 precision; -1e9 can become -inf in bf16
- **Use LR of 3e-5 with cosine schedule instead of 1e-4**: Docs recommend 3e-5 for 600M AR-to-dLLM adaptation; achieved better convergence (7.52 final loss vs 9.30 in previous run)
- **Increase block_size from 128 to 512 tokens on A100**: A100 has sufficient memory; larger blocks improve training stability and throughput (29.4 samples/sec achieved)
- **Create comprehensive RunPod setup documentation**: Capture gotchas (Docker images, venv issues, flash-attn build, trainer logging) to avoid repeating 20+ minute delays on future runs
