# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

The user wanted to: (1) review and improve the previous training setup, (2) deploy the model training to A100 GPU on RunPod with proper FA2 support, (3) complete a full 2000-step training run while logging metrics and checkpoints, (4) evaluate the trained model's effectiveness, and (5) plan a longer training run with potentially additional financial datasets. The strategy involved fixing architectural issues with FA2, setting up RunPod infrastructure correctly, monitoring training progress, and assessing model quality through text generation and perplexity metrics.

## Summary

The user and AI agent reviewed a previous diffusion language model training run on MPS, then conducted a full A100 GPU training run (2000 steps) on RunPod with corrected Flash Attention 2 support and improved hyperparameters, achieving a final loss of 7.52. The session included code fixes for FA2 bidirectional attention, RunPod deployment, comprehensive documentation updates, and evaluation of the trained model.

## Dead Ends

- **Using uv venv on RunPod with pre-built flash-attn wheels**: The isolated venv installed wrong PyTorch version (2.10.0+cu128 instead of 2.4.1+cu124), causing flash-attn source build to take 20-40 minutes and ultimately timeout
- **Building flash-attn from source on A100**: CUDA kernel compilation took 20-40 minutes for flash-attn 2.8.3, unnecessarily slow when system Python already had correct PyTorch+CUDA
- **Creating RunPod pod without SSH keys**: Pod was inaccessible for setup and monitoring; required deletion and recreation with SSH key authentication
- **Using wrong RunPod Docker image tags (latest, 1.13.1)**: Tags didn't exist or had missing dependencies; required identifying correct tag (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)
- **Using 4D mask approach for FA2 bidirectional attention**: Flash Attention 2 doesn't support 4D attention masks; required explicit is_causal=False flag on attention modules instead

## Decisions

- **Fixed FA2 bidirectional attention by setting is_causal=False on all attention modules when using flash_attention_2, while keeping 4D mask approach for eager/sdpa backends**: HF Transformers Flash Attention 2 implementation requires explicit is_causal parameter and ignores 4D masks; separate code paths handle different attention implementations
- **Used system Python directly on RunPod instead of creating isolated venv with uv**: RunPod's PyTorch Docker image already has PyTorch 2.4.1+cu124 properly configured; creating venv caused package version conflicts and slow builds
- **Corrected learning rate from 1e-4 to 3e-5 based on training-recipes.md recommendations for 600M model AR-to-dLLM adaptation**: Documentation specified 3e-5 LR; previous run used 1e-4 which was too aggressive for fine-tuning
- **Increased block size from 128 to 512 tokens for A100 training**: A100 has 80GB VRAM; 512-token blocks leverage GPU capacity better than MPS-conservative 128-token blocks
- **Added --attn_implementation flag to pretrain.py to allow switching between eager/sdpa/flash_attention_2**: Different GPU types benefit from different attention implementations; parameter enables runtime selection without code changes
