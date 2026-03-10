# Research: Training Recipes and Hyperparameters for Diffusion LLMs

## Executive Summary

Across all major dLLM papers (LLaDA, MDLM, Dream, Fast-dLLM v2, d1/diffu-GRPO), the optimizer is universally AdamW with standard betas. Peak learning rates range from 3e-4 (small models, from scratch) down to 1e-5 (7B+ adaptation). The dominant training paradigm has shifted from from-scratch pretraining to AR-to-dLLM adaptation, requiring only ~1-3B tokens at reduced LR (1-2e-5). Training stability is the primary unsolved risk: LLaDA crashed at 1.2T/2.3T tokens, requiring LR reduction from 4e-4 to 1e-4. MPS/Apple Silicon is viable for small-scale development but has hard limitations on Flash Attention and distributed training.

---

## Problem Statement

Building a dLLM training framework requires concrete hyperparameter choices across pretraining, adaptation, SFT, and RL (GRPO/DPO). Vague guidance like "use AdamW" is insufficient. This document extracts exact numbers from papers, codebases, and configs to provide copy-paste-ready training recipes for model sizes from 125M to 8B.

---

## Optimizer Choice

### Consensus: AdamW everywhere

Every major dLLM paper uses AdamW. No paper uses Adafactor, Lion, SOAP, or other alternatives.

| Paper | Optimizer | beta1 | beta2 | eps | Weight Decay |
|-------|-----------|-------|-------|-----|-------------|
| LLaDA 8B | AdamW | 0.9 (implied) | 0.999 (implied) | not stated | 0.1 |
| MDLM 110M | AdamW | 0.9 | 0.999 | default | 0.0 |
| Dream 7B | AdamW (implied) | not disclosed | not disclosed | not disclosed | not disclosed |
| Fast-dLLM v2 | AdamW | not stated | not stated | not stated | not stated |
| d1/diffu-GRPO | AdamW | 0.9 | 0.99 | default | 0.1 |

**Key observation:** MDLM uses zero weight decay — unusual but consistent across their experiments. LLaDA and d1 use 0.1 weight decay, which is standard for large Transformer training.

### Memory-efficient optimizers

No dLLM paper uses 8-bit Adam or paged optimizers. However, for single-GPU or memory-constrained setups, `bitsandbytes` 8-bit AdamW is a drop-in replacement that cuts optimizer state memory by 75% with <2% accuracy loss (validated on LLM training at scale via MLPerf 2025). The d1/diffu-GRPO codebase uses 4-bit QLoRA quantization for the base model, which achieves similar memory savings through a different mechanism.

**Recommendation for this framework:**
- Default: `torch.optim.AdamW` with beta1=0.9, beta2=0.999, weight_decay=0.1
- Memory-constrained: `bitsandbytes.optim.AdamW8bit` as opt-in flag
- For GRPO: beta2=0.99 (following d1), which provides slightly faster adaptation to reward signal changes

---

## Learning Rate Schedules

### From-scratch pretraining

| Paper | Model Size | Peak LR | Schedule | Warmup |
|-------|-----------|---------|----------|--------|
| LLaDA | 8B | 4e-4 | Warmup-Stable-Decay (WSD) | 2000 steps linear |
| MDLM | 110M | 3e-4 | Constant with warmup | not specified |
| BD3LM | 110M | 3e-4 (same codebase as MDLM) | Constant with warmup | not specified |

**LLaDA's WSD schedule in detail:**
1. Linear warmup: 0 -> 4e-4 over 2000 iterations
2. Stable phase: 4e-4 held for 1.2T tokens
3. Decay phase 1: Reduced to 1e-4 for 0.8T tokens (crash recovery)
4. Decay phase 2: Linear decay 1e-4 -> 1e-5 over final 0.3T tokens

The WSD schedule is notable because the "decay phase 1" was an emergency intervention after a NaN crash, not a planned schedule change. The intended schedule was likely WSD with a single decay phase.

### AR-to-dLLM adaptation

| Paper | Base Model | Peak LR | Tokens | Schedule |
|-------|-----------|---------|--------|----------|
| Fast-dLLM v2 | Qwen2.5-1.5B | 2e-5 | ~3.15B (6000 steps) | Linear warmup 500 steps |
| Fast-dLLM v2 | Qwen2.5-7B | 1e-5 | ~1.31B (2500 steps) | Linear warmup 500 steps |
| Dream 7B | Qwen2.5-7B | not disclosed | 580B | not disclosed |

**Critical insight:** Adaptation LR is 20-40x lower than from-scratch LR. Fast-dLLM v2 uses 1-2e-5 for adaptation vs 3-4e-4 for from-scratch. Dream's paper states "learning rate plays a critical role in preserving the beneficial properties inherited from AR initialization" but withholds the actual value.

### SFT

| Paper | Base | Peak LR | Schedule | Warmup | Epochs |
|-------|------|---------|----------|--------|--------|
| LLaDA SFT | LLaDA-8B-Base | 2.5e-5 | WSD | 50 steps linear | 3 epochs on 4.5M pairs |
| Dream SFT | Dream-7B-Base | 2e-6 | not specified | not specified | 3 epochs on 1.8M pairs |

**Dream SFT is 10x lower LR than LLaDA SFT.** This likely reflects Dream's use of CART (Context-Adaptive Reweighted Timestep) loss weighting, which changes the effective gradient magnitude.

### GRPO/RL

| Paper | Base | Peak LR | Schedule |
|-------|------|---------|----------|
| d1/diffu-GRPO | LLaDA-8B-Instruct | 3e-6 | Constant with warmup |

**Recommendation for this framework:**

| Stage | 125M | 600M | 1.3B | 8B |
|-------|------|------|------|-----|
| From-scratch pretrain | 3e-4 | 2e-4 | 1.5e-4 | 4e-4 (LLaDA) |
| AR adaptation | 5e-5 | 3e-5 | 2e-5 | 1e-5 |
| SFT | 5e-5 | 1e-5 | 5e-6 | 2e-6 |
| GRPO | 1e-5 | 5e-6 | 3e-6 | 3e-6 |

Schedule: WSD (warmup-stable-decay) for pretraining, cosine decay for SFT/GRPO. Warmup: 2000 steps for pretraining, 500 steps for adaptation, 50-100 steps for SFT/GRPO.

---

## Batch Size and Gradient Accumulation

| Paper | Model Size | Global Batch Size | Per-GPU Batch | Seq Length | Tokens/Batch |
|-------|-----------|-------------------|---------------|------------|-------------|
| LLaDA | 8B | 1280 | 4 | 4096 | 5.24M |
| MDLM | 110M | 512 | varies | 1024 | 524K |
| Fast-dLLM v2 | 1.5B/7B | 256 | varies | 2048 | 524K |
| Dream SFT | 7B | 8/GPU * 8 GPUs = 64 | 8 | 2048 | 131K |
| d1/diffu-GRPO | 8B | 12 (6 * 2 grad accum) | 6 | varies | varies |

**Scaling pattern:** ~500K tokens per batch is the sweet spot for models up to 1.5B. LLaDA scales to 5.24M tokens/batch at 8B, which is aggressive but consistent with AR LLM pretraining norms. GRPO uses tiny batch sizes (12) because each "sample" requires K=6 rollouts with 128 diffusion steps each.

**Recommendation:** Start with global_batch_size=512 for 125M, scale to 1024-2048 for 1.3B+. For GRPO, batch size is constrained by rollout memory — use gradient_accumulation_steps to compensate.

---

## AR-to-dLLM Adaptation Details

This is the most important section for practical training. From-scratch pretraining is largely obsolete for this framework.

### Fast-dLLM v2 recipe (most detailed, recommended starting point)

- **Base models:** Qwen2.5-1.5B, Qwen2.5-7B
- **Mechanism:** Block diffusion with complementary attention mask
- **Block size:** 32 (fixed), sub-block size 8
- **Optimizer:** AdamW
- **LR:** 2e-5 (1.5B), 1e-5 (7B)
- **Warmup:** 500 steps linear
- **Batch size:** 256 sequences * 2048 tokens = 524K tokens/step
- **Total tokens:** ~3.15B (1.5B model), ~1.31B (7B model)
- **Total steps:** 6000 (1.5B), 2500 (7B)
- **Hardware:** 64x A100, DeepSpeed Zero-3
- **Wall time:** ~8h (1.5B), ~12h (7B)

### Dream recipe (less detailed)

- **Base model:** Qwen2.5-7B (as Dream-v0-Base-7B)
- **Training corpus:** 580B tokens (NOT adaptation — full pretraining with AR init)
- **SFT data:** 1.8M instruction pairs from Tulu 3 + SmolLM 2, 3 epochs
- **SFT LR:** 2e-6
- **SFT per-GPU batch:** 8, micro-batch with gradient checkpointing
- **Loss weighting:** CART (context-adaptive reweighted timestep)

**Key distinction:** Dream trained on 580B tokens, which is NOT the ~1B token adaptation recipe. Dream uses AR initialization but then does extended pretraining. Fast-dLLM v2 is the true "adaptation in 1B tokens" approach.

### Recommended adaptation recipe for this framework

```yaml
# configs/training/adapt.yaml
optimizer: adamw
lr: 2e-5           # for 1.5B; use 1e-5 for 7B+
lr_schedule: linear_warmup_cosine_decay
warmup_steps: 500
weight_decay: 0.1
beta1: 0.9
beta2: 0.999
max_grad_norm: 1.0
batch_size: 256     # global, adjust per GPU count
seq_length: 2048
total_steps: 6000   # ~3B tokens for 1.5B model
bf16: true
gradient_checkpointing: true
```

---

## Training Stability

### LLaDA crash analysis

LLaDA crashed at 1.2T out of 2.3T tokens. The mitigation was:
1. Resume from checkpoint
2. Reduce LR from 4e-4 to 1e-4 (4x reduction)
3. Continue training at reduced LR for 0.8T tokens
4. Linear decay to 1e-5 over final 0.3T tokens

### ELBO-weighted loss stability

The ELBO weighting `ce / p_mask(t)` creates numerical instability near t=0 (where p_mask approaches 0). Mitigations:
- **Clamp p_mask:** `p_mask(t).clamp(min=1e-5)` — mandatory
- **Gradient clipping:** max_grad_norm=1.0 (LLaDA default, d1 uses 0.2 for GRPO)
- **NaN detection callback:** Reduce LR by 2x on first NaN, skip batch on subsequent NaN within window
- **Checkpoint frequency:** Every 1000 steps minimum

### d1/diffu-GRPO stability

d1 uses aggressive gradient clipping (max_grad_norm=0.2 vs 1.0 for pretraining). This is necessary because GRPO loss has high variance from MC sampling of the diffusion log-probabilities. Additional settings from d1:
- LoRA rank: 128, alpha: 64 (high rank for reasoning tasks)
- LoRA dropout: 0.05
- 4-bit base model quantization (QLoRA)
- K=6 rollouts per prompt
- 12 policy gradient inner update iterations per batch
- Remasking strategy: `low_confidence`

### Antithetic timestep sampling

Used by all papers for variance reduction. Implementation:
```python
u = torch.rand(1)
t = (u + torch.arange(B) / B) % 1.0  # antithetic samples
```
For DPO/GRPO: use SAME `u` for policy and reference model.

---

## MPS/Apple Silicon Considerations

### What works
- PyTorch MPS backend supports basic Transformer training
- `torch.nn.functional.scaled_dot_product_attention` works on MPS with some caveats
- Training small models (125M-600M) is viable for development/debugging
- bitsandbytes has experimental MPS support as of late 2025

### What does NOT work
- **Flash Attention 2:** Not available on MPS. The `flash-attn` package requires CUDA. Use `sdpa` attention implementation instead.
- **Distributed training:** MPS does not support DDP, FSDP, or DeepSpeed
- **Large sequence lengths:** SDPA on MPS has memory allocation failures above ~12K tokens
- **torch.compile:** Limited support on MPS, `fullgraph=True` mode not reliable
- **bf16:** MPS supports float16 but bf16 support is limited/absent on older chips

### Recommended MPS development setup
```python
# Use SDPA (not Flash Attention) on MPS
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    attn_implementation="sdpa",  # NOT "flash_attention_2"
    torch_dtype=torch.float16,   # NOT bf16 on MPS
)
# Explicit 4D mask approach works fine on MPS — no FA2 dependency
```

### MLX alternative
Apple's MLX framework outperforms PyTorch MPS by 2-3x for inference. However, MLX lacks HF Trainer integration, making it unsuitable for a framework built on HF Trainer. Use PyTorch MPS for development, target CUDA for production training.

---

## Recommended Default Stack

```toml
# pyproject.toml training defaults
[tool.diffusion_lm.training_defaults]
optimizer = "adamw"
lr = "3e-4"                    # from-scratch; override for adaptation/SFT
lr_schedule = "warmup_stable_decay"
warmup_steps = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999
max_grad_norm = 1.0
global_batch_size = 512
seq_length = 2048
bf16 = true
gradient_checkpointing = true
checkpoint_every_steps = 1000
```

---

## Open Questions

1. **Dream's pretraining LR:** Dream 7B trained on 580B tokens but withholds optimizer hyperparameters. Their SFT LR (2e-6) is 10x lower than LLaDA's (2.5e-5) — is this because of CART loss weighting or a fundamentally different training dynamic?

2. **Optimal adaptation token count:** Fast-dLLM v2 uses 1-3B tokens, but Dream uses 580B. Is the 500x compute savings real, or does Fast-dLLM v2's block diffusion approach require less adaptation by design?

3. **8-bit optimizer impact on ELBO loss:** No dLLM paper has tested 8-bit AdamW. The ELBO-weighted loss divides by small p_mask values — quantized optimizer states may interact poorly with these extreme gradient magnitudes. Needs empirical testing.

4. **MPS bf16 support timeline:** Apple has not committed to bf16 on MPS. This forces float16 on Apple Silicon, which may cause overflow issues with ELBO-weighted loss at scale.

5. **GRPO MC sample count:** d1 does not clearly document n_mc (number of Monte Carlo samples for log-probability estimation). The paper's Algorithm 2 references it but the config files only show K=6 rollouts.

---

## Sources

- [LLaDA: Large Language Diffusion Models (arXiv 2502.09992)](https://arxiv.org/abs/2502.09992)
- [MDLM: Simple and Effective Masked Diffusion Language Models (arXiv 2406.07524)](https://arxiv.org/abs/2406.07524)
- [MDLM GitHub Repository](https://github.com/kuleshov-group/mdlm)
- [Dream 7B: Diffusion Large Language Models (arXiv 2508.15487)](https://arxiv.org/abs/2508.15487)
- [Dream GitHub Repository](https://github.com/DreamLM/Dream)
- [d1: Scaling Reasoning in Diffusion LLMs via RL (arXiv 2504.12216)](https://arxiv.org/abs/2504.12216)
- [d1/diffu-GRPO GitHub Repository](https://github.com/dllm-reasoning/d1)
- [Fast-dLLM v2: Efficient Block-Diffusion LLM (arXiv 2509.26328)](https://arxiv.org/abs/2509.26328)
- [Fast-dLLM GitHub Repository](https://github.com/NVlabs/Fast-dLLM)
- [Mercury: Ultra-Fast Language Models Based on Diffusion (arXiv 2506.17298)](https://arxiv.org/abs/2506.17298)
- [BD3LM: Block Diffusion (arXiv 2503.09573)](https://arxiv.org/abs/2503.09573)
- [BD3LM GitHub Repository](https://github.com/kuleshov-group/bd3lms)
- [bitsandbytes 8-bit Optimizers](https://huggingface.co/docs/bitsandbytes/main/en/optimizers)
- [PyTorch MPS Backend Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [HuggingFace MPS Training Guide](https://huggingface.co/docs/transformers/v4.37.2/perf_train_special)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
