# Diffusion LLM Training Framework

A training framework for diffusion-based language models (dLLMs) supporting masked diffusion
(LLaDA/Mercury style), block diffusion (BD3LM), and continuous embedding diffusion. Wraps any
HuggingFace causal LM backbone with bidirectional attention via explicit 4D mask injection, then
adds a full training and inference pipeline. Supports AR-to-dLLM adaptation (recommended init
path: ~500x less compute than from scratch), SFT, DPO, GRPO, and generation via First-Hitting
Sampler (20x faster, theoretically correct).

---

## Commands

### Build / Lint
```bash
uv run ruff check . --fix && uv run ruff format .
```

### Test
```bash
uv run pytest tests/ -v
```

### Test (fast, no slow integration tests)
```bash
uv run pytest tests/ -v -m "not slow"
```

### Lint only (no fix)
```bash
uv run ruff check .
```

### Pre-commit (all checks)
```bash
uv run pre-commit run --all-files
```

### Smoke test (pretrain, AR-to-dLLM adaptation from GPT-2)
```bash
uv run python scripts/pretrain.py \
  --model_name_or_path gpt2 \
  --init_from_pretrained true \
  --process_type masked \
  --schedule_type linear \
  --max_steps 100 \
  --per_device_train_batch_size 4 \
  --output_dir ./checkpoints/smoke-test
```

### Generate (First-Hitting Sampler, default)
```bash
uv run python scripts/generate.py \
  --model_path ./checkpoints/smoke-test \
  --prompt "The meaning of life is" \
  --sampler first_hitting \
  --num_steps 64 \
  --max_new_tokens 128
```

### GRPO alignment (preferred over DPO for reasoning)
```bash
uv run python scripts/grpo.py \
  --model_path ./checkpoints/sft \
  --reward_model_path ./checkpoints/reward \
  --max_steps 500
```

---

## Current Build Phase

**Phase 0: Project Scaffold** — not started

Track progress in `docs/build-plan.md`. Do not advance to the next phase until the gate test for
the current phase passes.

---

## Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Required for modern type hint syntax (`X \| Y`, `Self`) |
| uv | latest | Package management, virtual environments |
| PyTorch | >=2.2 | Tensor ops, autograd, distributed |
| HuggingFace Transformers | >=4.40,<5.10 | Backbone models, Trainer, tokenizers — pin upper bound |
| HuggingFace datasets | >=2.18 | Streaming dataset loading |
| flash-attn | >=2.5 | FA2 kernel — `is_causal=False` passed explicitly |
| accelerate | >=0.28 | FSDP/DeepSpeed integration |
| peft | >=0.10 | LoRA adapters |
| lm-evaluation-harness | >=0.4 | Benchmarks |
| wandb | >=0.16 | Experiment tracking |
| loguru | >=0.7 | Structured logging |
| ruff | >=0.4 | Linting and formatting |
| pytest | >=8.0 | Testing |

---

## Key Architectural Decisions — Do Not Revisit

These decisions are final. Do not second-guess them when implementing.

1. **Use HF Trainer, not a custom training loop.** `DiffusionTrainer` extends `HFTrainer` and
   overrides only `compute_loss()`. All gradient accumulation, mixed precision, distributed
   training, checkpointing, and logging are inherited.

2. **Bidirectional attention via explicit 4D mask injection — NOT monkey-patch.**
   `_update_causal_mask` is deprecated and removed in Transformers v5.10. The new
   `masking_utils.create_causal_mask` path breaks `torch.compile(fullgraph=True)` (issue #42950).
   The Gemma3 bidirectional mask bug (issue #39389) proves silent causal fallback is a real failure
   mode — no crash, just wrong attention and degraded quality.
   **Correct approach:** `BidirectionalTransformer.forward()` builds an explicit all-zeros 4D float
   mask `(B, 1, L, L)` and passes it directly. Bypasses all HF internals. FA2: `is_causal=False`
   passed explicitly. Do NOT use `AutoModelForMaskedLM` — smaller ecosystem.

3. **AR-to-dLLM adaptation as the default init path.** `ModelConfig.init_from_pretrained=True`
   by default. Start from a pretrained AR checkpoint, fine-tune with masked diffusion objective
   on ~1B tokens. ~500x less compute than training from scratch. Validated by Dream 7B (Aug 2025)
   and Fast-dLLM v2 (Oct 2025).

4. **First-Hitting Sampler as the default sampler (not confidence-based categorical).**
   Standard categorical sampling exploits a mathematical inaccuracy (arXiv:2409.02908) that
   inflates benchmarks. First-Hitting Sampler is theoretically correct, 20x faster, drop-in
   replacement. Assigns each masked token an independent first-hitting time from the score
   function, reveals in order. Default: `GenerationConfig.sampler = "first_hitting"`.

5. **No timestep input to the model (masked diffusion).** RADD paper proves this is theoretically
   unnecessary. Omitting it means any HF CausalLM can be used as-is. `ContinuousDiffusionLM` is
   the sole exception.

6. **ELBO-weighted loss is mandatory.** Dividing CE loss by `p_mask(t)` is required for correct
   training — not optional. Without it the model is undertrained at low-noise timesteps.

7. **Antithetic timestep sampling for every batch.** `t_i = (u + i/B) mod 1`. For DPO/GRPO, use
   the SAME timestep draw for both policy and reference model.

8. **GRPO preferred over DPO for reasoning tasks.** diffu-GRPO (arXiv:2504.12216) is the first
   working RL pipeline for dLLMs. LLaDA-8B + diffu-GRPO matches AR reasoning models on
   GSM8K/MATH500. Use `DiffusionGRPOTrainer` for reasoning alignment.

9. **src layout.** All importable code under `src/diffusion_lm/`. Scripts in `scripts/` are entry
   points only. Tests in `tests/` import from `diffusion_lm` (installed via `uv pip install -e .`).

---

## Critical Implementation Gotchas

### Causal mask — use explicit 4D mask, NOT monkey-patch
```python
# WRONG — deprecated, removed in v5.10, breaks torch.compile:
self.transformer.model._update_causal_mask = lambda *a, **kw: None

# CORRECT — explicit 4D zeros mask bypasses all HF mask generation:
def forward(self, input_ids, attention_mask=None, **kwargs):
    B, L = input_ids.shape
    if attention_mask is None or attention_mask.dim() != 4:
        # All-zeros float mask = no masking = full bidirectional attention
        attention_mask = torch.zeros(
            B, 1, L, L, dtype=self.transformer.dtype, device=input_ids.device
        )
    return self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
```
Verify bidirectionality: changing token at position 5 must change logits at position 3.

### Flash Attention 2 — bidirectional attention setup
FA2 does NOT support arbitrary 4D attention masks. For bidirectional attention with FA2:
1. Set `is_causal=False` on all attention modules in `__init__` (matches LLaDA/Dream approach)
2. Pass only a 2D padding mask (or None) in forward — NOT a 4D mask
3. If a 4D mask arrives (e.g. block-diagonal for packed seqs), log a warning and fall back to None

```python
# In __init__: patch all attention modules
if config.attn_implementation == "flash_attention_2":
    for module in self.transformer.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False

# In forward: FA2 path uses 2D mask only
if self._use_fa2:
    if attention_mask is not None and attention_mask.dim() == 4:
        attention_mask = None  # 4D not supported with FA2
else:
    # eager/sdpa path: build explicit 4D zeros mask
    attention_mask = torch.zeros(B, 1, L, L, dtype=dtype, device=device)
```
**Validated:** FA2 + is_causal=False on 28 attention modules with Qwen3-0.6B on A100.

### Loss normalization
```python
# Correct:
ce = F.cross_entropy(logits.view(-1, V), x0.view(-1), reduction='none').view(x0.shape)
weighted = ce / p_mask(t).unsqueeze(-1).clamp(min=1e-5)
loss = (weighted * masked_positions).sum() / masked_positions.sum().clamp(min=1)

# Wrong (undertrains near-clean steps):
loss = ce[masked_positions].mean()  # no ELBO weighting
```

### SFT prompt protection — apply BEFORE backbone, AFTER forward_process
```python
corrupted, mask = diffusion.forward_process(input_ids, t)
corrupted = torch.where(prompt_mask, input_ids, corrupted)  # restore prompt tokens
logits = backbone(corrupted, attention_mask)
loss = diffusion.compute_loss(logits, input_ids, corrupted, t, loss_mask=~prompt_mask)
```

### Cross-sample attention leakage in packed sequences
When packing multiple sequences into one batch row, the collator MUST emit a per-sample
block-diagonal attention mask. Without it, tokens from different samples attend to each other —
silent quality degradation with no crash signal.

### "Answer Backslide" prevention
9.8% of MATH-500 failures have correct intermediate tokens overwritten in later steps.
Enable `running_confidence_remasking=True` in `GenerationConfig` (default). Free, no retraining.

### DPO/GRPO memory requirements
```python
# DPO: ~32 forward passes per step (4 quantities × 8 MC samples)
# GRPO: K=8 rollouts + n_mc=8 per rollout = 64+ forward passes
# Defaults for large models:
training_args = DiffusionTrainingArguments(
    gradient_checkpointing=True,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
# DPO: cpu_offload_ref=True for reference model
# GRPO: process rollouts sequentially if OOM
```

### DPO/GRPO antithetic sampling for variance reduction
Use the SAME antithetic timestep `u` for both policy model and reference model when estimating
a single pair. Their log-prob difference has much lower variance with shared timesteps.

### EOS suppression during generation
```python
# At non-final denoising positions:
logits[:, :, eos_token_id] = float('-inf')  # before argmax
```

### Continuous diffusion rounding
- **Training:** straight-through estimator — `x_rounded = x + (x_rounded - x).detach()`
- **Inference:** exact nearest-neighbor — `torch.cdist(predicted_emb, embedding_table).argmin(-1)`

### mask_token_id must be set before training
`DiffusionConfig.mask_token_id` defaults to `-1` (invalid). `DiffusionTrainer.__init__` must
raise `ValueError` if still `-1`. Always call `add_mask_token(model, tokenizer)` first.

### LogLinear schedule numerical stability
```python
alpha = torch.exp(-(-torch.log1p(-(1 - eps) * t))).clamp(0.0, 1.0)
```

### NaN crashes at scale
LLaDA crashed at 1.2T/2.3T tokens without mitigation. Defaults:
- `max_grad_norm=1.0` (clip gradient norm)
- Checkpoint every 1000 steps
- LR reduction at first NaN (handled by `NanLossCallback`)

### Transformers version pinning
Pin `transformers<5.10` in `pyproject.toml`. The removal of `_update_causal_mask` in v5.10
does not affect our explicit 4D mask approach, but other internal changes may. Bump only
after explicit compatibility testing. Validated with transformers 5.3.0 on A100.

### Padding mask value
Use `torch.finfo(dtype).min` for padding mask values, NOT hardcoded `-1e9`. The correct value
depends on dtype (bf16 has different range than fp32). This prevents silent attention corruption.

---

## RunPod GPU Training

See `docs/runpod-setup.md` for full setup guide. Key gotchas:

1. **Docker image**: Use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Many tags
   that look valid don't exist.
2. **DO NOT use `uv venv`** — it installs PyTorch from PyPI with wrong CUDA version. Use system
   pip: `pip install -e ".[dev]"`
3. **flash-attn**: Use `pip install flash-attn` (finds prebuilt wheels). `uv pip install` triggers
   a 20-40 min source build.
4. **SSH key**: Set via `PUBLIC_KEY` env var on pod creation.
5. **Loss logging**: HF Trainer tqdm overwrites loss values in redirected logs. Check
   `trainer_state.json` in checkpoints for loss history.

### Validated A100 training command
```bash
python3 scripts/pretrain.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --process_type masked --schedule_type linear \
  --block_size 512 --dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --dataset_name ashraq/financial-news-articles \
  --text_column text --dataset_split train \
  --output_dir /workspace/checkpoints/finance-qwen3-a100 \
  --max_steps 2000 --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 --bf16 true \
  --learning_rate 3e-5 --lr_scheduler_type cosine \
  --warmup_steps 200 --weight_decay 0.1 \
  --max_grad_norm 1.0 --logging_steps 25 \
  --save_steps 500 --save_total_limit 4 \
  --report_to none --gradient_checkpointing true \
  --dataloader_num_workers 4
```

---

## File Structure

```
diffusion/
├── CLAUDE.md                          # This file
├── pyproject.toml                     # Single config: deps, ruff, pytest
├── .env.example                       # Template for secrets
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── README.md
│
├── src/
│   └── diffusion_lm/
│       ├── __init__.py
│       ├── config/
│       │   ├── model.py               # ModelConfig (init_from_pretrained=True default)
│       │   ├── diffusion.py           # DiffusionConfig (process_type, block_size)
│       │   ├── training.py            # DiffusionTrainingArguments (DPO + GRPO params)
│       │   └── generation.py         # GenerationConfig (sampler="first_hitting" default)
│       ├── schedules/
│       │   ├── base.py               # NoiseSchedule ABC
│       │   ├── linear.py             # alpha(t) = 1 - t
│       │   ├── cosine.py             # alpha(t) = cos(pi/2 * t)
│       │   └── loglinear.py          # alpha(t) = exp(-sigma(t))
│       ├── diffusion/
│       │   ├── base.py               # DiffusionProcess ABC
│       │   ├── masked.py             # MaskedDiffusionProcess
│       │   ├── block.py              # BlockDiffusionProcess (BD3LM, arXiv:2503.09573)
│       │   └── continuous.py         # ContinuousDiffusionProcess
│       ├── models/
│       │   ├── backbone.py           # BidirectionalTransformer (explicit 4D mask)
│       │   ├── masked_diffusion_lm.py
│       │   └── continuous_diffusion_lm.py
│       ├── data/
│       │   ├── pretraining.py        # PretrainingDataset
│       │   ├── sft.py                # SFTDataset
│       │   ├── preference.py         # PreferenceDataset
│       │   └── collators.py          # Collators with block-diagonal attention masks
│       ├── trainers/
│       │   ├── base.py               # DiffusionTrainer(HFTrainer) + NanLossCallback
│       │   ├── pretraining.py        # PretrainingTrainer
│       │   ├── sft.py                # SFTTrainer
│       │   ├── dpo.py                # DiffusionDPOTrainer
│       │   └── grpo.py               # DiffusionGRPOTrainer (diffu-GRPO)
│       ├── samplers/
│       │   ├── base.py               # Sampler ABC, SamplerOutput
│       │   ├── first_hitting_sampler.py  # ★ Default. 20x faster, theoretically correct
│       │   ├── block_sampler.py      # BD3LM sampler with KV cache
│       │   ├── continuous_sampler.py # DDPM denoising + rounding
│       │   └── cached_sampler.py     # Cached predictions (~2-3x speedup)
│       └── evaluation/
│           ├── perplexity.py         # ELBOPerplexity
│           └── lm_eval_adapter.py    # DiffusionLMEvalAdapter
│
├── scripts/
│   ├── pretrain.py
│   ├── sft.py
│   ├── dpo.py
│   ├── grpo.py                        # diffu-GRPO alignment
│   ├── generate.py
│   └── evaluate.py
│
├── configs/
│   ├── models/
│   │   ├── small.yaml                # ~125M (GPT-2 scale)
│   │   ├── medium.yaml               # ~350M
│   │   ├── large.yaml                # ~1.3B
│   │   └── 8b.yaml                   # ~8B (LLaMA-3 scale, AR-to-dLLM target)
│   ├── diffusion/
│   │   ├── masked_linear.yaml
│   │   ├── masked_cosine.yaml
│   │   ├── block_linear.yaml         # BD3LM config
│   │   └── continuous_cosine.yaml
│   └── training/
│       ├── pretrain.yaml
│       ├── sft.yaml
│       ├── dpo.yaml
│       └── grpo.yaml
│
└── tests/
    ├── conftest.py                    # Shared fixtures
    ├── test_schedules.py
    ├── test_diffusion.py
    ├── test_models.py
    ├── test_samplers.py
    └── test_trainers.py
```

---

## Reference Implementations

When in doubt, consult these before inventing solutions:

- LLaDA (masked diffusion, 8B, MIT): https://github.com/ML-GSAI/LLaDA
- dLLM unified library (Feb 2026): https://github.com/ZHZisZZ/dllm
- BD3LM block diffusion (ICLR 2025 Oral): https://github.com/kuleshov-group/bd3lms
- d1 diffu-GRPO (RL for dLLMs): https://github.com/dllm-reasoning/d1
- MDLM (NeurIPS 2024, log-linear schedule): https://github.com/kuleshov-group/mdlm
- Mercury 1 paper (generation speed): https://arxiv.org/abs/2506.17298
- First-Hitting Sampler paper: https://arxiv.org/abs/2409.02908
- Fast-dLLM (NVlabs, inference accel): https://github.com/NVlabs/Fast-dLLM — verify license
