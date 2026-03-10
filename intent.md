# Intent

ok, first review the previous training run that we did and the settings we used

## Interpreted Goal

The user wanted to deploy the diffusion LM model to production GPU hardware (A100 on RunPod), fix Flash Attention 2 compatibility issues for bidirectional attention, run a full 2000-step pretraining run on financial data with proper logging and checkpointing, and then evaluate the trained model's generation quality. The strategy involved fixing the FA2 implementation in the backbone, correcting the setup scripts for RunPod's environment, and implementing comprehensive evaluation (generation + perplexity).

## Summary

Set up and executed a complete A100 GPU training run on RunPod for diffusion language model pretraining, achieving a final loss of 7.52 over 2000 steps in 36 minutes ($0.90), followed by initial model evaluation.

## Dead Ends

- **Using uv venv on RunPod with pre-installed PyTorch**: uv created an isolated venv that reinstalled PyTorch 2.10.0+cu128 instead of using the system's pre-installed PyTorch 2.4.1+cu124, causing version mismatches and flash-attn to build from source (20-40 min compilation)
- **Using wrong RunPod PyTorch Docker image tags**: Initial image tags (e.g., `runpod/pytorch:2.5.0`) didn't exist in the registry, causing pod initialization failures
- **Relying on 4D mask for Flash Attention 2 bidirectional attention**: FA2 doesn't support 4D attention masks; requires explicit `is_causal=False` flag set on attention modules at initialization

## Decisions

- **Use system Python directly on RunPod instead of uv venv**: RunPod's PyTorch image comes with pre-installed PyTorch and CUDA properly configured; uv venv isolation caused redundant PyTorch reinstalls and broke version consistency
- **Set is_causal=False on attention modules during FA2 initialization in BidirectionalTransformer**: FA2 explicitly requires is_causal=False for bidirectional attention; the 4D mask approach works for eager/sdpa but not FA2, so FA2 detection at init time ensures proper attention computation
- **Add --attn_implementation flag to pretrain.py CLI args**: Allows runtime control over attention backend (eager/sdpa/flash_attention_2) without code modification, enabling A100 optimization
- **Lower learning rate from 1e-4 to 3e-5 per training docs for 600M AR adaptation**: Training recipes recommend 3e-5 for 600M model AR-to-dLLM adaptation; previous MPS run used 1e-4 which caused higher loss (9.30 at step 500)
- **Increase block size from 128 to 512 tokens for A100**: Previous run used MPS-conservative block size of 128; A100 has 80GB VRAM and can comfortably handle 512-token blocks, improving throughput
