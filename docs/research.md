# Research: Memory-Efficient Training on Apple Silicon (MPS Backend)

## Executive Summary

Training a ~600M parameter model (Qwen3-0.6B) on Apple Silicon is feasible but requires careful configuration. **bitsandbytes 8-bit Adam does NOT work on MPS** — use Adafactor or AdamW with gradient checkpointing instead. bf16 autocast works on MPS starting PyTorch >=2.6 with macOS >=14. On 18GB unified memory, expect batch size 1-2 with seq_len 512 and gradient checkpointing; 36GB allows batch size 4-8. torch.compile provides no meaningful benefit on MPS — skip it. The most impactful optimizations are: (1) gradient checkpointing, (2) Adafactor instead of AdamW, (3) fp16 or bf16 mixed precision, (4) aggressive gradient accumulation.

## Problem Statement

We need to train/fine-tune a ~600M parameter diffusion LM on Apple Silicon Macs (M-series, 18-36GB unified memory) using PyTorch's MPS backend and HuggingFace Trainer. The key constraints are limited memory (shared between CPU and GPU), immature MPS backend, and missing CUDA-specific optimizations.

## Technology Evaluation

### 1. Memory-Efficient Optimizers on MPS

#### Adafactor — Recommended
- **Status:** Fully works on MPS. Pure PyTorch, no custom kernels.
- **Memory savings:** ~33% less optimizer state than AdamW (factored second moments instead of per-parameter).
- **HF Trainer flag:** `--optim adafactor`
- **Gotcha:** Set `scale_parameter=False` and provide explicit LR when using with Trainer. Default Adafactor uses relative step sizing which can conflict with HF schedulers.

#### AdamW (torch) — Recommended (baseline)
- **Status:** Fully works on MPS.
- **HF Trainer flag:** `--optim adamw_torch`
- **Memory:** Full optimizer state (2x model params). For 600M model: ~4.8GB optimizer state in fp32.

#### Schedule-Free AdamW — Consider
- **Status:** Pure Python/PyTorch, should work on MPS. Requires `pip install schedulefree`.
- **HF Trainer flag:** `--optim schedule_free_adamw` (with `--lr_scheduler_type constant`)
- **Benefit:** Eliminates LR scheduler tuning. Competitive with tuned cosine schedules.

#### bitsandbytes 8-bit Adam — Avoid on MPS
- **Status:** Does NOT work on MPS as of bitsandbytes 0.45.x. CUDA-only for quantized optimizers. Apple Silicon support is listed as "experimental/alpha" but quantized optimizer kernels are not implemented for Metal.
- **HF Trainer flag:** `--optim adamw_bnb_8bit` (will crash on MPS)
- **Alternative:** The `mps-bitsandbytes` fork on PyPI offers NF4/INT8 quantization for inference but NOT optimizer quantization for training.

#### Lion — Consider (manual integration)
- **Status:** Pure PyTorch implementation available (`lion-pytorch` package). Works on MPS since it only uses sign operations.
- **HF Trainer flag:** Not built-in. Requires custom optimizer injection via `Trainer.__init__(optimizers=(lion_opt, scheduler))`.
- **Memory:** Same as SGD (only momentum, no second moments). ~50% less than AdamW.
- **Gotcha:** Requires 3-10x lower LR than AdamW. Sensitive to weight decay.

#### CAME — Avoid
- **Status:** External package `came-pytorch`. Should work on MPS (pure PyTorch) but poorly maintained, last update 2023.

#### GaLore — Consider for extreme memory savings
- **Status:** `--optim galore_adamw`. Projects gradients to low-rank space.
- **Memory:** Can reduce optimizer memory by 8x with rank=64.
- **Gotcha:** Adds ~3 min startup overhead for SVD. Single-GPU only for layerwise mode.

### 2. Gradient Checkpointing on MPS

**Status: Works. Use it.**
- HF Trainer flag: `--gradient_checkpointing true`
- Memory savings: ~40-60% reduction in activation memory, at ~20-30% training speed cost.
- **No known MPS-specific issues.** The implementation is backend-agnostic (pure autograd).
- For a 600M model at seq_len 512, this is the single most impactful memory optimization.

### 3. Mixed Precision on MPS

#### bf16 — Recommended (PyTorch >=2.6, macOS >=14)
- `torch.autocast("mps", dtype=torch.bfloat16)` merged in PyTorch via PR #139390 (Nov 2024). Available in PyTorch 2.6+.
- HF Trainer flag: `--bf16 true`
- **Requires macOS 14 (Sonoma) or later.** Will raise an error on older macOS.
- Preferred over fp16: no loss scaling needed, wider dynamic range.

#### fp16 — Consider (with caveats)
- `torch.autocast("mps", dtype=torch.float16)` available since PyTorch 2.1.
- HF Trainer flag: `--fp16 true`
- **Gotcha:** HuggingFace docs previously stated "MPS does not support fp16." This referred to *native* fp16 training, not autocast. Autocast fp16 works but may require `PYTORCH_ENABLE_MPS_FALLBACK=1` for some ops.
- Risk of overflow/underflow without proper loss scaling.

#### fp32 — Fallback
- Always works. 2x memory cost. Use only if mixed precision causes instability.

### 4. Gradient Accumulation on MPS

**Works, but with caveats.**
- HF Trainer flag: `--gradient_accumulation_steps N`
- **Performance concern:** MPS backward passes are more fragmented than CUDA. Each accumulation step incurs Metal command buffer overhead. Expect ~5-15% overhead per accumulation step vs. theoretical zero-cost.
- **Bug alert:** PyTorch MPS had a kernel bug where `addcmul_` and `addcdiv_` silently failed on non-contiguous tensors (affects gradient accumulation scenarios). Fixed in PyTorch 2.3+. Pin `torch>=2.3`.
- **Recommendation:** Use accumulation steps of 4-16. Effective batch size = micro_batch * accumulation_steps.

### 5. Practical Memory Budgets

Memory breakdown for Qwen3-0.6B (~600M params):

| Component | fp32 | bf16/fp16 mixed |
|-----------|------|-----------------|
| Model weights | ~2.4 GB | ~1.2 GB |
| Optimizer state (AdamW) | ~4.8 GB | ~4.8 GB (always fp32) |
| Optimizer state (Adafactor) | ~3.2 GB | ~3.2 GB |
| Gradients | ~2.4 GB | ~1.2 GB |
| Activations (bs=1, seq=512) | ~1-2 GB | ~0.5-1 GB |
| Activations (bs=1, seq=512, grad ckpt) | ~0.3-0.5 GB | ~0.2-0.3 GB |
| PyTorch/system overhead | ~1-2 GB | ~1-2 GB |

**18GB Mac (e.g., M3/M4 base):**
- Usable GPU memory: ~13-14 GB (Metal caps at ~75% by default)
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to use full memory (risk of system instability)
- **Max config:** bf16 + Adafactor + gradient_checkpointing + batch_size=1 + seq_len=512 + grad_accum=16
- Effective batch size: 16. Tight but workable.
- For seq_len=1024: likely OOM. Reduce to batch_size=1 with grad_accum=32.

**36GB Mac (e.g., M3/M4 Pro):**
- Usable GPU memory: ~27-30 GB
- **Max config:** bf16 + AdamW + gradient_checkpointing + batch_size=4 + seq_len=512 + grad_accum=8
- Effective batch size: 32. Comfortable.
- seq_len=1024 feasible with batch_size=2.

### 6. torch.compile on MPS

**Status: Avoid for training.**
- MPS backend lacks a mature compiler/fusion stack. Complex operations fall back to CPU or run as unfused Metal kernels.
- Most users run eager mode on MPS. No meaningful speedup reported for training workloads.
- **Risk:** Graph breaks are common in training (backward pass fragmentation, gradient checkpointing recomputation), each triggering recompilation overhead.
- **Alternative for speed:** MLX framework (Apple's native ML framework) is 2-3x faster than PyTorch MPS for inference, but not compatible with HF Trainer for training.

### 7. MPS-Specific Bugs and Workarounds

| Issue | PyTorch Version | Workaround |
|-------|----------------|------------|
| `addcmul_`/`addcdiv_` silent failure on non-contiguous tensors | <2.3 | Upgrade to >=2.3 |
| Missing ops fallback to CPU silently | All | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| OOM without warning | All | Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` |
| SDPA crashes at seq_len >12K | All | Use `attn_implementation="eager"` for long seqs |
| MPS sometimes slower than CPU for small models | All | Profile first; MPS overhead dominates for <100M models |
| Memory leak in LSTM iterations | <2.5 | Not relevant for transformer training |
| MPS not available on macOS 26 Tahoe (beta) | 2.9.1/2.10 nightly | Known issue, tracked in pytorch/pytorch#167679 |

**Critical env vars for MPS training:**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # or 0.7 for safety
```

## Recommended Stack

For Qwen3-0.6B on 18GB Apple Silicon Mac:

```python
# TrainingArguments
TrainingArguments(
    output_dir="./checkpoints",
    bf16=True,                          # requires macOS >=14, PyTorch >=2.6
    optim="adafactor",                  # 33% less memory than AdamW
    gradient_checkpointing=True,        # 40-60% activation memory savings
    per_device_train_batch_size=1,      # tight on 18GB
    gradient_accumulation_steps=16,     # effective batch = 16
    max_grad_norm=1.0,
    dataloader_pin_memory=False,        # not useful on unified memory
    # Do NOT set: fp16=True (use bf16), torch_compile=True (no benefit)
)
```

For 36GB Mac, change `per_device_train_batch_size=4` and `gradient_accumulation_steps=8`.

**Minimum pinned versions:**
```
torch>=2.6
transformers>=4.40,<5.10
accelerate>=0.28
```

## Open Questions

1. **PyTorch 2.6+ bf16 autocast stability on MPS in practice** — merged Nov 2024 but limited community reports on training stability for large models. Test with a short run first.
2. **Schedule-Free optimizer on MPS** — theoretically works (pure PyTorch) but no MPS-specific benchmarks found.
3. **LoRA + Adafactor on MPS** — PEFT LoRA should work on MPS but the combination with Adafactor is undertested. If memory is critical, this could reduce trainable params by 10-100x.
4. **macOS 26 (Tahoe) compatibility** — PyTorch MPS is currently broken on macOS 26 beta (issue #167679). Monitor before upgrading.

## Sources

- [bitsandbytes Apple Silicon support — Issue #252](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/252)
- [bitsandbytes multi-backend discussion — Discussion #1340](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340)
- [PyTorch MPS bf16 autocast — Issue #139386](https://github.com/pytorch/pytorch/issues/139386)
- [PyTorch MPS bf16 autocast PR — PR #139390](https://github.com/pytorch/pytorch/pull/139390)
- [PyTorch MPS fp16 autocast — PR #99272](https://github.com/pytorch/pytorch/pull/99272)
- [PyTorch MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
- [HuggingFace Apple Silicon training docs](https://huggingface.co/docs/transformers/en/perf_train_special)
- [HuggingFace Accelerate MPS guide](https://huggingface.co/docs/accelerate/en/usage_guides/mps)
- [HuggingFace Optimizers documentation](https://huggingface.co/docs/transformers/en/optimizers)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
- [Profiling Apple Silicon Performance for ML Training (arXiv:2501.14925)](https://arxiv.org/pdf/2501.14925)
- [MPS addcmul_ bug — blog post](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)
- [PyTorch MPS broken on macOS 26 — Issue #167679](https://github.com/pytorch/pytorch/issues/167679)
- [MPS SDPA memory issues — Medium](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)
- [mps-bitsandbytes PyPI](https://www.piwheels.org/project/mps-bitsandbytes/)
- [bitsandbytes v0.45.2 installation guide](https://huggingface.co/docs/bitsandbytes/v0.45.2/installation)
