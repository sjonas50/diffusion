# Architecture: Diffusion LLM Training Framework

## System Overview

A training framework for diffusion-based language models (dLLMs) supporting masked diffusion
(LLaDA/Mercury style), block diffusion (BD3LM), and continuous embedding diffusion. The system
wraps any HuggingFace causal LM backbone, injects an explicit bidirectional attention mask to
enable full bidirectional attention, then surrounds it with a diffusion process that corrupts
inputs during training and iteratively denoises during inference. The recommended init path is
**AR-to-dLLM adaptation** (fine-tuning a pretrained AR checkpoint with ~1B diffusion tokens),
which requires ~500x less compute than training from scratch.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PATHS                                     │
│                                                                             │
│  ┌──────────────┐    ┌───────────────────┐    ┌─────────────────────────┐  │
│  │  Pretrain    │    │   SFT Data        │    │   Preference Data       │  │
│  │  Data        │    │  (prompt_mask)    │    │  (chosen / rejected)    │  │
│  │  Streaming   │    │                   │    │                         │  │
│  └──────┬───────┘    └────────┬──────────┘    └───────────┬─────────────┘  │
│         │                    │                            │                 │
│         ▼                    ▼                            ▼                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Collators                                     │   │
│  │     RandomTruncateCollator / SFTCollator / DPOCollator               │   │
│  │     (all collators emit per-sample block-diagonal attention masks    │   │
│  │      when packing sequences — prevents cross-sample attention leak)  │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────────────┐
│                          DIFFUSION CORE                                      │
│                                  │                                           │
│                    ┌─────────────▼──────────────┐                           │
│                    │      Noise Schedules        │                           │
│                    │  LinearSchedule             │                           │
│                    │  CosineSchedule             │                           │
│                    │  LogLinearSchedule          │                           │
│                    │  alpha(t), weight(t)        │                           │
│                    └─────────────┬──────────────┘                           │
│                                  │                                           │
│      ┌───────────────────────────┼──────────────────────────┐               │
│      ▼                           │                           ▼               │
│  ┌──────────────────┐            │          ┌────────────────────────┐       │
│  │ MaskedDiffusion  │   ┌────────▼──────┐   │ ContinuousDiffusion    │       │
│  │ Process          │   │ BlockDiffusion│   │ Process                │       │
│  │ forward_process  │   │ Process       │   │ forward_process        │       │
│  │ compute_loss     │   │ (BD3LM)       │   │ compute_loss           │       │
│  └────────┬─────────┘   └──────┬────────┘   └───────────┬────────────┘       │
│           │                    │                         │                    │
└───────────┼────────────────────┼─────────────────────────┼────────────────────┘
            │                    │                         │
┌───────────┼────────────────────┼─────────────────────────┼────────────────────┐
│           │              MODEL LAYER                     │                    │
│           ▼                    │                         ▼                    │
│  ┌──────────────────┐          │          ┌────────────────────────┐          │
│  │ MaskedDiffusionLM│          │          │ ContinuousDiffusionLM  │          │
│  └────────┬─────────┘          │          └───────────┬────────────┘          │
│           │                    │                      │                       │
│           └────────────────────┼──────────────────────┘                       │
│                                ▼                                              │
│                    ┌──────────────────────────┐                               │
│                    │  BidirectionalTransformer │                               │
│                    │  (any HF CausalLM)        │                               │
│                    │  explicit 4D float mask   │                               │
│                    │  injected in forward()    │                               │
│                    │  bypasses all HF internals│                               │
│                    └──────────────────────────┘                               │
└──────────────────────────────────────────────────────────────────────────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────────┐          ┌───────────────────────────┐
│     TRAINERS          │          │       SAMPLERS            │
│                       │          │                           │
│  DiffusionTrainer     │          │  FirstHittingSampler ★    │
│  PretrainingTrainer   │          │  BlockSampler (BD3LM)     │
│  SFTTrainer           │          │  ContinuousSampler        │
│  DiffusionDPOTrainer  │          │  CachedSampler            │
│  DiffusionGRPOTrainer │          │                           │
└───────────────────────┘          └───────────────────────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────────┐          ┌───────────────────────────┐
│     EVALUATION        │          │       SCRIPTS             │
│                       │          │                           │
│  ELBOPerplexity       │          │  pretrain.py              │
│  LMEvalAdapter        │          │  sft.py / dpo.py          │
└───────────────────────┘          │  grpo.py                  │
                                   │  generate.py              │
                                   │  evaluate.py              │
                                   └───────────────────────────┘

★ = default sampler
```

---

## Components

| Component | Purpose | Key Classes | Inputs / Outputs |
|-----------|---------|-------------|-----------------|
| `config/` | Typed configuration for every subsystem | `ModelConfig`, `DiffusionConfig`, `DiffusionTrainingArguments`, `GenerationConfig` | In: YAML/CLI args. Out: frozen dataclasses passed to all other components |
| `schedules/` | Noise schedule math: masking probability at time t | `LinearSchedule`, `CosineSchedule`, `LogLinearSchedule` | In: `t ∈ [0,1]`. Out: `alpha(t)` (keep prob), `weight(t)` for loss weighting |
| `diffusion/masked.py` | Forward corruption and ELBO-weighted CE loss for masked diffusion | `MaskedDiffusionProcess` | In: `(x0, t)`. Out: `(x_t, mask)` forward; `loss` scalar backward |
| `diffusion/block.py` | BD3LM block diffusion: AR over blocks, masked within | `BlockDiffusionProcess` | In: `(x0, t, block_size)`. Out: block-masked `x_t`, block-structured loss |
| `diffusion/continuous.py` | Gaussian noise over embeddings, MSE denoising loss | `ContinuousDiffusionProcess` | In: `(x0_emb, t)`. Out: `(x_t_emb, noise)`, MSE `loss` |
| `models/backbone.py` | Bidirectional wrapper around any HF CausalLM via explicit 4D mask injection | `BidirectionalTransformer` | In: `(input_ids, attention_mask)`. Out: `logits (B, L, V)` |
| `models/masked_diffusion_lm.py` | Full masked diffusion model: sample t, corrupt, denoise, loss | `MaskedDiffusionLM` | In: `(input_ids, prompt_mask?)`. Out: `{"loss", "logits"}` |
| `models/continuous_diffusion_lm.py` | Continuous embedding diffusion with rounding head | `ContinuousDiffusionLM` | In: `(input_ids, t?)`. Out: `{"loss", "logits"}` |
| `data/pretraining.py` | Streaming tokenize-and-group | `PretrainingDataset` | In: HF dataset name. Out: fixed-length `input_ids` chunks |
| `data/sft.py` | Prompt/response formatting with `prompt_mask` | `SFTDataset` | In: `{prompt, response}` or `{messages}`. Out: `(input_ids, prompt_mask)` |
| `data/preference.py` | Chosen/rejected pairs for DPO/GRPO | `PreferenceDataset` | In: `{prompt, chosen, rejected}`. Out: `(chosen_ids, rejected_ids, prompt_mask)` |
| `data/collators.py` | Padding, truncation, per-sample attention masks for packed sequences | `RandomTruncateCollator`, `SFTCollator`, `DPOCollator` | In: list of samples. Out: padded batch tensors + block-diagonal attention mask |
| `trainers/base.py` | HF Trainer extension with diffusion loss hook | `DiffusionTrainer` | In: model + dataset. Out: checkpoint, metrics |
| `trainers/pretraining.py` | Adds random truncation (1% of batches) | `PretrainingTrainer` | Extends `DiffusionTrainer` |
| `trainers/sft.py` | Enforces prompt protection via `prompt_mask` | `SFTTrainer` | Extends `DiffusionTrainer` |
| `trainers/dpo.py` | ELBO-based DPO with frozen reference model | `DiffusionDPOTrainer` | In: policy + ref model + preference batch. Out: DPO loss |
| `trainers/grpo.py` | diffu-GRPO RL pipeline (arXiv:2504.12216) | `DiffusionGRPOTrainer` | In: policy model + reward model + prompts. Out: GRPO loss |
| `samplers/first_hitting_sampler.py` | **Default.** Theoretically correct, 20x faster than categorical. Supports Running Confidence Remasking | `FirstHittingSampler` | In: `(model, prompt_ids, gen_config)`. Out: `SamplerOutput` |
| `samplers/block_sampler.py` | BD3LM block sampler — AR over blocks with KV cache | `BlockSampler` | In: same as above. Out: `SamplerOutput` with KV cache speedup |
| `samplers/continuous_sampler.py` | DDPM reverse process + nearest-neighbor token rounding | `ContinuousSampler` | In: same. Out: `SamplerOutput` |
| `samplers/cached_sampler.py` | Caches stable predictions across steps (~2-3x speedup) | `CachedSampler` | In: same. Out: `SamplerOutput` |
| `evaluation/perplexity.py` | ELBO-based perplexity upper bound | `ELBOPerplexity` | In: model + dataset. Out: `{"ppl_bound", "nll_per_token"}` |
| `evaluation/lm_eval_adapter.py` | lm-evaluation-harness interface | `DiffusionLMEvalAdapter` | In: `lm_eval` requests. Out: log-likelihoods / generated text |

---

## Data Flow: Pretraining (AR-to-dLLM Adaptation — Recommended)

```
0. Load pretrained AR checkpoint (e.g. LLaMA-3-8B, Qwen2.5-7B)
      └─► BidirectionalTransformer wraps it, injects 4D bidirectional mask
      └─► No weight changes — diffusion training starts immediately

1. HuggingFace datasets (streaming=True)
      └─► PretrainingDataset.tokenize_and_group()
               chunks text into fixed-length sequences (e.g. 2048 tokens)

2. RandomTruncateCollator
      └─► with probability 1%, truncate batch to random shorter length
      └─► for packed batches: emit per-sample block-diagonal attention mask
               prevents cross-sample attention leakage (silent quality degradation)
      └─► produces: {input_ids, attention_mask}

3. PretrainingTrainer.compute_loss()
      └─► antithetic timestep sampling: t_i = (u + i/B) mod 1, scaled to [eps, 1)
      └─► MaskedDiffusionProcess.forward_process(input_ids, t)
               returns: (corrupted_ids, token_mask)
      └─► BidirectionalTransformer(corrupted_ids, attention_mask)
               forward() injects explicit zeros 4D mask (B, 1, L, L)
               returns: logits (B, L, V)
      └─► MaskedDiffusionProcess.compute_loss(logits, input_ids, corrupted_ids, t)
               CE(logits[masked], targets[masked]) / p_mask(t)  -- ELBO weighting
               returns: scalar loss

4. HF Trainer handles:
      backward pass, gradient accumulation, mixed precision,
      optimizer step, LR schedule, logging to W&B, checkpointing
      clip grad norm to 1.0 — NaN crashes observed at 1.2T tokens without it
```

## Data Flow: SFT

```
1. SFTDataset
      └─► tokenizes prompt + response concatenated
      └─► produces prompt_mask: True at prompt positions, False at response positions

2. SFTCollator
      └─► pads to batch max length, preserves prompt_mask alignment

3. SFTTrainer.compute_loss()
      └─► MaskedDiffusionProcess.forward_process(input_ids, t)
      └─► apply prompt protection BEFORE backbone:
               corrupted = where(prompt_mask, input_ids, corrupted)
               (prompt tokens never get masked)
      └─► BidirectionalTransformer(corrupted_ids, attention_mask)
      └─► MaskedDiffusionProcess.compute_loss(..., loss_mask=~prompt_mask)
               loss only computed over response token positions
               normalized by response token count (not total sequence length)
```

## Data Flow: GRPO (Reasoning Alignment — Preferred over DPO)

```
1. Prompt dataset → DiffusionGRPOTrainer

2. For each prompt, sample K=8 completions using FirstHittingSampler
      └─► each completion scored by reward model

3. Compute group-relative advantage:
      A_k = (r_k - mean(r)) / std(r)  per group

4. Estimate log p(completion | prompt) via ELBO:
      for n_mc_samples=8:
          sample shared t (antithetic) for all K completions
          loss_k += model(concat(prompt, completion_k), prompt_mask)["loss"]
      log_prob_k = -mean(loss_k)

5. GRPO loss = -mean(A_k * log_prob_k) with PPO-style clip (ratio ∈ [1-ε, 1+ε])
```

## Data Flow: DPO (Preference Alignment)

```
1. PreferenceDataset  ->  DPOCollator
      └─► produces: chosen_input_ids, rejected_input_ids, prompt_mask

2. DiffusionDPOTrainer.compute_loss()
      └─► for n_mc_samples=8 with SHARED antithetic t for policy AND reference:
               policy_chosen_loss   += model(chosen_input_ids, prompt_mask)["loss"]
               policy_rejected_loss += model(rejected_input_ids, prompt_mask)["loss"]
      └─► with torch.no_grad():
               ref_chosen_loss    += ref_model(chosen_input_ids, ...)["loss"]
               ref_rejected_loss  += ref_model(rejected_input_ids, ...)["loss"]
      └─► log_probs = -mean_loss_over_mc_samples
      └─► dpo_loss = -log_sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))
```

## Data Flow: Generation (First-Hitting Sampler — Default)

```
1. scripts/generate.py  ->  FirstHittingSampler.generate(model, prompt_ids, gen_config)

2. Initialize:
      x = [prompt_ids | MASK * gen_len]

3. Forward pass → score function s(x, t) for all masked positions

4. Sample "first-hitting time" τ_i for each masked token i
      from distribution parameterized by score function

5. Reveal tokens in order of τ_i (smallest τ first)
      Optional: Running Confidence Remasking
        - if revealed token confidence < threshold, allow re-masking
        - prevents "Answer Backslide" (9.8% MATH-500 failure rate without this)
      Suppress EOS at non-final positions (set logit to -inf before argmax)

6. Return: SamplerOutput(sequences=x, scores=confidence_history)

Note: No iterative remask/unmask loop — one pass determines reveal order.
      20x faster than confidence-based categorical sampler.
      Theoretically correct: categorical sampling exploits a mathematical inaccuracy.
```

---

## External Dependencies

| Dependency | Version | Role | Auth | Risk / Notes |
|------------|---------|------|------|-------------|
| `transformers` | >=4.40, <5.10 | Backbone models, Trainer, tokenizers | none | `_update_causal_mask` removed in v5.10 — pin upper bound. New `masking_utils.create_causal_mask` breaks `torch.compile(fullgraph=True)` (issue #42950) |
| `datasets` | >=2.18 | Streaming dataset loading | `HF_TOKEN` for gated datasets | Network failure with streaming; wrap in retry with backoff |
| `torch` | >=2.2 | Tensor ops, autograd, distributed | none | Clip grad norm to 1.0; NaN crashes observed at scale without it |
| `flash-attn` | >=2.5 | FA2 kernel — pass `is_causal=False` explicitly | none | Build fails on non-CUDA; graceful fallback to SDPA |
| `lm-evaluation-harness` | >=0.4 | Benchmark eval | none | Pin version; task API changes between releases |
| `wandb` | >=0.16 | Experiment tracking | `WANDB_API_KEY` | Offline mode if key missing; non-fatal |
| `peft` | >=0.10 | LoRA adapters | none | Optional; only if `ModelConfig.use_lora=True` |
| `accelerate` | >=0.28 | FSDP/DeepSpeed via HF Trainer | none | Config mismatch crashes at Trainer init |
| Fast-dLLM (NVlabs) | N/A | Inference acceleration reference | none | **NVIDIA Research license — verify commercial rights before incorporating** |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For gated models | HuggingFace API token for LLaMA, Mistral, etc. |
| `WANDB_API_KEY` | No | W&B logging. Training continues without it (logs to stdout) |
| `WANDB_PROJECT` | No | W&B project name. Defaults to `diffusion-lm` |
| `WANDB_RUN_NAME` | No | Override auto-generated run name |
| `HF_DATASETS_CACHE` | No | Override default datasets cache directory |
| `TRANSFORMERS_CACHE` | No | Override default model cache directory |
| `CUDA_VISIBLE_DEVICES` | No | GPU selection for single-node multi-GPU |
| `TOKENIZERS_PARALLELISM` | No | Set to `false` to suppress HF tokenizer fork warnings |

---

## Key Architectural Decisions

### 1. Extend HF Trainer rather than build a custom training loop
HF Trainer provides gradient accumulation, mixed precision (bf16/fp16), DDP/FSDP/DeepSpeed distributed training, checkpoint management, and W&B logging out of the box. The only customization point needed is `compute_loss()`, which is a clean override.

### 2. Bidirectional attention via explicit 4D float mask injection (NOT monkey-patch)
`_update_causal_mask` is deprecated and removed in Transformers v5.10. The new `masking_utils.create_causal_mask` path breaks `torch.compile(fullgraph=True)` (open issue #42950). The Gemma3 bug (issue #39389) demonstrates that silent causal fallback is a real failure mode — model trains with wrong attention, no crash signal, just degraded quality.

**Correct approach:** `BidirectionalTransformer.forward()` injects an explicit all-zeros 4D float mask `(batch, 1, seq_len, seq_len)` directly. This bypasses all HF internal mask generation machinery and is stable across Transformers versions. For Flash Attention 2, `is_causal=False` is passed explicitly to `flash_attn_func` — not inferred from config.

### 3. AR-to-dLLM adaptation as the preferred initialization path
Starting from a pretrained AR checkpoint (LLaMA-3, Qwen2.5, etc.) and fine-tuning with the masked diffusion objective on ~1B tokens requires ~500x less compute than training from scratch. Dream 7B (Aug 2025) and Fast-dLLM v2 (Oct 2025) both validate this approach. `ModelConfig.init_from_pretrained=True` is the default.

### 4. First-Hitting Sampler as the default sampler (not confidence-based categorical)
Standard categorical sampling in masked diffusion exploits a mathematical inaccuracy (proven in arXiv:2409.02908) that inflates benchmark scores compared to AR models. The First-Hitting Sampler is theoretically correct, 20x faster, and a drop-in replacement. It assigns each masked token an independent "first-hitting time" from the score-parameterized distribution and reveals tokens in hitting-time order — no iterative remask/unmask loop.

### 5. No timestep input to the model (masked diffusion)
The RADD paper proves conditioning a masked diffusion model on `t` is theoretically unnecessary: the model infers noise level from mask token density. Omitting `t` means any HF CausalLM can be used as-is with zero architectural changes. `ContinuousDiffusionLM` is the sole exception — it uses a lightweight timestep embedding because noise level is not observable from the embedding input.

### 6. ELBO-weighted loss is mandatory
Dividing per-token cross-entropy by `p_mask(t)` is not optional. Without it, the model is undertrained at low-noise timesteps (the critical final denoising steps). This is required for correct training — not an optimization.

### 7. Antithetic timestep sampling for every batch
For a batch of B sequences: draw one `u ~ Uniform[0,1)`, set `t_i = (u + i/B) mod 1`. This stratified sampler ensures the full noise range is covered every batch, reducing gradient variance by ~B-fold. For DPO/GRPO, use the SAME timestep draw for policy and reference model to reduce variance of their difference.

---

## Scaling Considerations

| Bottleneck | Breaks At | Mitigation |
|------------|-----------|------------|
| GPU memory (pretraining) | ~7B params on single A100 80GB bf16 | FSDP ZeRO-3 via `accelerate`; gradient checkpointing |
| GPU memory (DPO) | ~3B params (policy + frozen ref + 8 MC samples) | CPU offload for reference model; `gradient_checkpointing=True` |
| GPU memory (GRPO) | ~7B params (policy + K=8 rollouts in memory) | Reduce K; process rollouts sequentially; gradient checkpointing |
| Gradient variance | All scales | Antithetic timestep sampling; MIRROR masks for extreme cases |
| NaN crashes | >1T tokens | Clip grad norm to 1.0; checkpoint every 1k steps; LR reduction |
| Cross-sample attention leakage | Packed sequences | Per-sample block-diagonal mask in all collators |
| Generation latency | >512 tokens | FirstHittingSampler (20x vs categorical); BlockSampler (KV cache) |
| lm-eval cost | N/A (eval-only) | 32 MC samples per loglikelihood is expensive — reduce for rapid iteration |
