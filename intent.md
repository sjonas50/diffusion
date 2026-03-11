# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

The user wanted to review previous training configurations, prepare the codebase for A100 GPU execution with Flash Attention 2 optimization, deploy to RunPod cloud infrastructure, monitor the training run, document the setup process, and then evaluate the trained model's effectiveness using text generation and perplexity metrics.

## Summary

The session involved setting up and executing a 2000-step diffusion language model training run on a RunPod A100 GPU, fixing Flash Attention 2 bidirectional attention implementation, and conducting three post-training evaluations to assess model quality.

## Dead Ends

- **Creating RunPod pod with incorrect Docker image tag (runpod/pytorch:latest)**: Tag didn't exist in Docker Hub; required correct versioned tag like runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- **Using 'uv venv' for Python environment on RunPod**: Created isolated venv that reinstalled PyTorch 2.10.0 (wrong CUDA version), breaking compatibility with system's pre-configured PyTorch 2.4.1+cu124; added unnecessary dependency management overhead
- **Building flash-attn from source on RunPod**: CUDA kernel compilation took 20-40 minutes; switched to using system pip to install prebuilt wheel instead
- **Using 4D mask approach with Flash Attention 2**: FA2 doesn't support arbitrary attention masks; requires explicit is_causal=False flag set on attention modules instead

## Decisions

- **Fixed Flash Attention 2 support by detecting FA2 in __init__ and setting is_causal=False on all attention modules, while keeping 4D mask approach for eager/sdpa backends**: Flash Attention 2 has different API requirements than eager/sdpa; requires explicit bidirectional flag rather than mask-based signaling; matches approach used in LLaDA and Dream papers
- **Added --attn_implementation CLI flag to pretrain.py to allow runtime selection between eager, sdpa, and flash_attention_2**: Enables users to experiment with different attention implementations without code changes; necessary for GPU type portability
- **Used system Python directly on RunPod instead of creating venv**: RunPod's PyTorch image comes pre-configured with correct CUDA/PyTorch versions; venv isolation was counterproductive and caused version conflicts
- **Lowered learning rate from 1e-4 to 3e-5 (as per docs recommendations)**: Previous MPS run with 1e-4 showed plateauing at loss 9.3; docs recommend 3e-5 for 600M AR-to-dLLM adaptation; new run achieved better convergence (7.52 final loss)
- **Extended block_size from 128 to 512 tokens for A100**: A100 has sufficient memory (80GB); larger blocks improve tokenization efficiency and context modeling; MPS constraint (128) was unnecessary for cloud GPU
- **Prioritized documentation over optimization: created runpod-setup.md, updated CLAUDE.md and README with learned gotchas**: Flash Attention 2 bidirectional setup and RunPod deployment patterns are non-obvious; documenting the specific issues (is_causal flag, Docker image tags, venv problems) saves future debugging time
