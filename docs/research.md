# Diffusion LLM Research — March 2026

## Executive Summary

Diffusion LLMs have matured significantly since mid-2025. Mercury 2 (Feb 24, 2026) confirms the approach works commercially at 727–1196 tok/s with quality competitive with Claude 4.5 Haiku. The dominant training paradigm has shifted from **training from scratch** to **AR-to-dLLM adaptation** — starting from a pretrained autoregressive checkpoint requires ~500x less compute. Four new techniques materially improve on the original LLaDA plan: First-Hitting Sampler (20x faster inference, drop-in), Block Diffusion/BD3LM (KV cache + 13% perplexity gain, ICLR 2025 Oral), diffu-GRPO (first working RL pipeline), and CDLM distillation (14x post-training speedup). The HuggingFace `_update_causal_mask` monkey-patch approach is **deprecated and being removed in v5.10** — must be replaced with explicit 4D attention mask injection.

---

## Problem Statement

Build a training framework for diffusion-based LLMs supporting masked diffusion (LLaDA/Mercury style) and continuous embedding diffusion. Target: full pipeline from AR-adapted pretraining through SFT, DPO/RL, and generation with state-of-the-art inference speedups.

---

## Technology Evaluation — Frameworks

| Rank | Framework | Stars | Status | Use For |
|------|-----------|-------|--------|---------|
| 1 | [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) | 2.1k | Active Feb 2026, paper arXiv:2602.22661 | **Primary reference — unified training** |
| 2 | [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | ~800 | ICLR 2025 Oral | Block diffusion, KV cache, noise schedules |
| 3 | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | ~3k | Active, LLaDA-1.5 + LLaDA-V released | Reference for SFT + fine-tuning patterns |
| 4 | [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | N/A | Oct 2025, ICLR 2026 | Inference acceleration — **verify NVIDIA Research license before use** |
| 5 | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | N/A | Apr 2025, arXiv:2504.12216 | diffu-GRPO RL fine-tuning |

---

## Mercury 2 — Verified Facts (Feb 24, 2026)

- **Speed:** 727 tok/s (Artificial Analysis measured), peak 1,196 tok/s on Blackwell — 13x faster than Claude 4.5 Haiku, 17x faster than GPT-5 Mini
- **TTFT caveat:** 3.48s first-token latency (2x median peers) — all diffusion steps must complete before first token streams
- **Quality:** AIME 2025: 91.1 | GPQA Diamond: 73.6 | LiveCodeBench: 67.3 | AI Intelligence Index: 33/134
- **Tier:** Claude 4.5 Haiku / GPT-5 Mini quality range. Co-founder confirmed: not GPT-4o class.
- **Context:** 128K tokens, OpenAI-compatible API
- **Pricing:** $0.25/M input, $0.75/M output. Note: generates 3.5x more output tokens than peers — real cost is higher than headline rate suggests.
- **Architecture:** Discrete masked diffusion. Speed comes from custom CUDA kernels for parallel inference + Blackwell hardware, not architectural novelty.
- **Weights:** Closed, proprietary, API-only.

---

## New Models Since June 2025

| Model | Date | Size | Key Innovation |
|-------|------|------|----------------|
| LLaDA-V | May 2025 | 8B | Multimodal vision-language (SigLIP2 vision tower) |
| Dream 7B | Aug 2025 | 7B | AR-to-dLLM init, context-adaptive noise scheduling |
| MMaDA-8B | Jun 2025 | 8B | Block diffusion + mixed CoT + UniGRPO RL, text+image |
| LLaDA-1.5 | 2025 | 8B | Updated instruction-tuned checkpoint (open weights, MIT) |
| Tiny-A2D | Dec 2025 | 0.5–0.6B | SOTA small dLLMs via AR adaptation |
| LLaDA 2.0-flash | Dec 2025 | 100B MoE | 73.18 avg over 47 benchmarks, ties Qwen3-30B-A3B; weights release TBD |

**Key paradigm shift:** AR-to-dLLM adaptation is now standard. ~500x less compute vs. training from scratch. Dream 7B and Fast-dLLM v2 both demonstrate this.

---

## Architecture Patterns Found

### 1. AR-to-dLLM Adaptation (New Standard)
Start from pretrained AR checkpoint → disable causal mask correctly → fine-tune with masked diffusion objective on ~1B tokens. Fast-dLLM v2 full recipe. Preferred init path.

### 2. Block Diffusion — BD3LM (ICLR 2025 Oral, arXiv:2503.09573)
Autoregressive over blocks, masked diffusion within each block. Unlocks KV caching, arbitrary sequence length. 13% perplexity improvement. MMaDA-8B uses this approach.

### 3. First-Hitting Sampler (arXiv:2409.02908, ICLR 2025)
Proves timestep `t` adds no information to MDM score function (confirms no-timestep-input). Standard categorical sampling exploits a mathematical inaccuracy that inflates benchmarks. First-Hitting Sampler is a **drop-in replacement** with 20x faster inference and theoretically correct samples. Must be default sampler.

### 4. diffu-GRPO / d1 (arXiv:2504.12216)
First working RL pipeline for dLLMs. Adapts GRPO for parallel token generation. LLaDA-8B + diffu-GRPO matches AR reasoning models on GSM8K/MATH500. Replaces DPO for reasoning tasks.

### 5. Consistency Distillation — CDLM (arXiv:2511.19269)
Post-training distillation giving 14x inference speedup. Together AI published full recipe. Optional Phase 6.

### 6. Running Confidence Remasking
Inference-time fix for "Answer Backslide." Free, no retraining. Implement in sampler as default.

---

## Key APIs and Dependencies

| Dependency | Status | Risk | Notes |
|-----------|--------|------|-------|
| HuggingFace Transformers | Active | **HIGH** | `_update_causal_mask` deprecated, removal in v5.10 |
| Flash Attention 2 | Active | Medium | Must pass `is_causal=False` explicitly to `flash_attn_func` |
| `torch.compile` | Active | Medium | Fails with new `masking_utils.create_causal_mask` (issue #42950) |
| HF datasets streaming | Active | Low | Stable API |
| lm-evaluation-harness | Active | Low | Interface stable |
| Fast-dLLM (NVlabs) | Active | **License** | NVIDIA Research license — verify commercial rights before use |

---

## Known Pitfalls and Risks

### CRITICAL: Replace the Causal Mask Monkey-Patch
`_update_causal_mask` is deprecated and removed in Transformers v5.10. The new `masking_utils.create_causal_mask` path breaks `torch.compile(fullgraph=True)`. The Gemma3 bidirectional mask bug (issue #39389) shows silent causal fallback is a real failure mode — model trains on wrong attention with no crash signal.

**Safe path:** Subclass model's `forward()` and inject an explicit all-zeros 4D float mask `(batch, 1, seq_len, seq_len)`. Bypasses all HF internal mask generation. For FA2: pass `is_causal=False` explicitly to `flash_attn_func`.

### Training Pitfalls (ranked by severity)
1. **"Answer Backslide"** — 9.8% of MATH-500 failures had correct intermediate answers overwritten. Mitigate: Running Confidence Remasking (free) or MDPO RL.
2. **Gradient variance** — Random masking causes 14x higher variance than full-mask. Mitigate: MIRROR (anti-correlated mask pairs) or P-POTS sampler.
3. **NaN crashes at scale** — LLaDA crashed at 1.2T/2.3T tokens; LR reduction fixed it. Mitigate: clip grad norm to 1.0, checkpoint every 1k steps.
4. **Cross-sample attention leakage** — Silent quality degradation when packing sequences. Mitigate: per-sample block-diagonal attention masks in collator.

### Benchmark Inflation Warning
Standard categorical sampling in masked diffusion exploits a mathematical inaccuracy that inflates benchmark scores vs. autoregressive models. Use First-Hitting Sampler for honest comparisons.

---

## Recommended Stack

```
Core:         PyTorch 2.5+ + HuggingFace Transformers (pin version) + uv
Diffusion:    Masked diffusion primary, continuous secondary
Init:         AR-to-dLLM adaptation from pretrained checkpoint (not from scratch)
Attention:    Explicit 4D float mask injection in forward() — NOT monkey-patch
Sampler:      First-Hitting Sampler as default (not confidence-based categorical)
RL:           diffu-GRPO (better than DPO for reasoning tasks)
Inference:    Running Confidence Remasking (free), CDLM distillation (optional Phase 6)
Architecture: BD3LM block diffusion variant for KV cache + long sequences
Training:     HF Trainer extension (still correct — gives DDP/FSDP/DeepSpeed free)
Config:       HF-style dataclasses + HfArgumentParser
```

---

## Required Updates to plan.md

1. **Replace `_update_causal_mask` monkey-patch** → explicit 4D mask injection in `BidirectionalTransformer.forward()`
2. **Add AR-to-dLLM adaptation** as preferred init path in `ModelConfig`
3. **Replace masked sampler with First-Hitting Sampler** as default
4. **Add Running Confidence Remasking** option to sampler
5. **Add diffu-GRPO trainer** alongside/replacing DPO trainer
6. **Add Block Diffusion (BD3LM)** variant in `diffusion/` module
7. **Update Mercury 2 facts**: 727–1196 tok/s, TTFT 3.48s, AIME 91.1, Feb 2026
8. **Pin Transformers version** in pyproject.toml
9. **Add cross-sample attention mask** in data collators for packed sequences
10. **Update references**: add arXiv:2409.02908, 2503.09573, 2504.12216, 2511.19269, 2602.22661

---

## Open Questions

- Will LLaDA 2.0-flash (100B MoE) release open weights? Would be the best base model for adaptation.
- Does diffu-GRPO work at <8B scale with limited compute?
- Optimal block size for BD3LM across different hardware configurations?
- Does CDLM distillation preserve reasoning quality or only benefit throughput tasks?

---

## Sources

- [Mercury 2 Launch — Inception Labs](https://www.inceptionlabs.ai/blog/introducing-mercury-2)
- [Mercury 2 — Artificial Analysis](https://artificialanalysis.ai/models/mercury-2)
- [Mercury 1 Paper — arXiv:2506.17298](https://arxiv.org/abs/2506.17298)
- [dLLM Unified Framework — arXiv:2602.22661](https://arxiv.org/abs/2602.22661)
- [ZHZisZZ/dllm GitHub](https://github.com/ZHZisZZ/dllm)
- [Time-Agnostic MDMs / First-Hitting Sampler — arXiv:2409.02908](https://arxiv.org/abs/2409.02908)
- [BD3LM Block Diffusion — arXiv:2503.09573](https://arxiv.org/abs/2503.09573)
- [kuleshov-group/bd3lms GitHub](https://github.com/kuleshov-group/bd3lms)
- [Fast-dLLM — arXiv:2505.22618](https://arxiv.org/abs/2505.22618)
- [NVlabs/Fast-dLLM GitHub](https://github.com/NVlabs/Fast-dLLM)
- [d1 diffu-GRPO — arXiv:2504.12216](https://arxiv.org/abs/2504.12216)
- [CDLM Distillation — arXiv:2511.19269](https://arxiv.org/pdf/2511.19269)
- [Dream 7B — arXiv:2508.15487](https://arxiv.org/html/2508.15487v1)
- [LLaDA — arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- [MMaDA — arXiv:2505.15809](https://arxiv.org/abs/2505.15809)
- [MDLM — arXiv:2406.07524](https://arxiv.org/abs/2406.07524)
