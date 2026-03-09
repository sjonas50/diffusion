# Build Plan: Diffusion LLM Training Framework

## Phase 0: Project Scaffold
**Goal:** Establish a reproducible, linted, tested project skeleton before writing any logic.

- [ ] Create `pyproject.toml` with `[project]`, `[project.optional-dependencies]`, `[tool.ruff]`, and `[tool.pytest.ini_options]` sections. Pin all major dependencies including `transformers<5.10` upper bound. `[S]`
  - File: `/Users/sjonas/diffusion/pyproject.toml`
- [ ] Create `src/` layout: `diffusion_lm/__init__.py` and stub `__init__.py` files for all subpackages (`config`, `models`, `diffusion`, `schedules`, `samplers`, `trainers`, `data`, `evaluation`). `[S]`
  - Directory: `/Users/sjonas/diffusion/src/diffusion_lm/`
- [ ] Create `tests/conftest.py` with shared fixtures: `tiny_tokenizer`, `tiny_gpt2_config`, `device`. `[S]`
  - File: `/Users/sjonas/diffusion/tests/conftest.py`
- [ ] Create empty test files: `tests/test_schedules.py`, `tests/test_diffusion.py`, `tests/test_models.py`, `tests/test_samplers.py`, `tests/test_trainers.py`. Each must contain at least one placeholder test (`assert True`). `[S]`
- [ ] Create `.env.example` with all required variables: `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_RUN_NAME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `CUDA_VISIBLE_DEVICES`, `TOKENIZERS_PARALLELISM`. `[S]`
  - File: `/Users/sjonas/diffusion/.env.example`
- [ ] Create `.gitignore` covering `.env`, `__pycache__`, `*.egg-info`, `checkpoints/`, `.ruff_cache`, `.pytest_cache`, `dist/`. `[S]`
- [ ] Create `.pre-commit-config.yaml` with ruff-check and ruff-format hooks. `[S]`
- [ ] Create `scripts/` directory with stub scripts: `pretrain.py`, `sft.py`, `dpo.py`, `grpo.py`, `generate.py`, `evaluate.py`. Each prints `"Not yet implemented"` and exits 0. `[S]`
- [ ] Create `configs/` directory with subdirectories `models/`, `diffusion/`, `training/` and at minimum `configs/models/small.yaml`. `[S]`

**Gate:** `uv run ruff check . && uv run pytest tests/ -q --co` passes (collection only, all placeholders pass).

---

## Phase 1: Core Math — Configs, Schedules, Diffusion Processes
**Goal:** Build the mathematical foundation. All downstream components depend on this being correct.

### 1a. Configuration dataclasses
- [ ] `src/diffusion_lm/config/model.py` — `ModelConfig`: `model_name_or_path: str`, `init_from_pretrained: bool = True` (AR-to-dLLM adaptation is the default), `attn_implementation: str = "flash_attention_2"`, `dtype: str = "bfloat16"`, `use_lora: bool = False`, `lora_rank: int = 16`, `lora_alpha: float = 32.0`. `[S]`
- [ ] `src/diffusion_lm/config/diffusion.py` — `DiffusionConfig`: `process_type: Literal["masked", "block", "continuous"]`, `schedule_type: Literal["linear", "cosine", "loglinear"] = "linear"`, `mask_token_id: int = -1`, `time_epsilon: float = 1e-3`, `block_size: int | None = None` (for BD3LM). `[S]`
- [ ] `src/diffusion_lm/config/training.py` — `DiffusionTrainingArguments` extends HF `TrainingArguments`: adds `random_truncation_ratio: float = 0.01`, `dpo_beta: float = 0.1`, `dpo_num_mc_samples: int = 8`, `grpo_group_size: int = 8`, `grpo_clip_ratio: float = 0.2`, `grpo_num_mc_samples: int = 8`. `[S]`
- [ ] `src/diffusion_lm/config/generation.py` — `GenerationConfig`: `num_steps: int = 64`, `temperature: float = 1.0`, `sampler: Literal["first_hitting", "block", "continuous", "cached"] = "first_hitting"`, `block_size: int | None = None`, `remasking: Literal["confidence", "random"] = "confidence"`, `running_confidence_remasking: bool = True`, `max_new_tokens: int = 128`, `guidance_scale: float = 0.0`. `[S]`

### 1b. Noise schedules
- [ ] `src/diffusion_lm/schedules/base.py` — abstract `NoiseSchedule` with abstract `alpha(t: Tensor) -> Tensor`, `alpha_derivative(t)`, concrete `weight(t) = -alpha_derivative(t) / (1 - alpha(t)).clamp(min=1e-5)`, `mask_probability(t) = 1 - alpha(t)`. `[S]`
- [ ] `src/diffusion_lm/schedules/linear.py` — `LinearSchedule`: `alpha(t) = 1 - t`. `[S]`
- [ ] `src/diffusion_lm/schedules/cosine.py` — `CosineSchedule`: `alpha(t) = cos(pi/2 * t)`. `[S]`
- [ ] `src/diffusion_lm/schedules/loglinear.py` — `LogLinearSchedule` (MDLM): `alpha(t) = exp(-(-log(1 - (1 - eps) * t)))`, `eps = 1e-4`. Clamp output to `[0, 1]`. `[M]`

### 1c. Diffusion processes
- [ ] `src/diffusion_lm/diffusion/base.py` — abstract `DiffusionProcess`: abstract `forward_process`, `compute_loss`, `sample_timesteps`. `[S]`
- [ ] `src/diffusion_lm/diffusion/masked.py` — `MaskedDiffusionProcess`: `[M]`
  - `sample_timesteps(B, device)`: antithetic — draw `u ~ U[0,1)`, `t_i = (u + i/B) mod 1`, scale to `[time_epsilon, 1)`.
  - `forward_process(x0, t)`: `p_mask = 1 - schedule.alpha(t)` broadcast to `(B, L)`, independent Bernoulli mask, replace masked with `mask_token_id`.
  - `compute_loss(logits, x0, corrupted, t, loss_mask=None)`: CE over masked positions, weighted by `1 / p_mask(t).clamp(min=1e-5)`, normalized by `masked.sum().clamp(min=1)`.
- [ ] `src/diffusion_lm/diffusion/block.py` — `BlockDiffusionProcess` (BD3LM, arXiv:2503.09573): `[M]`
  - Autoregressive over blocks of size `block_size`, masked diffusion within each block.
  - `forward_process(x0, t, block_boundaries)`: mask independently per-block using per-block `t`.
  - `compute_loss(...)`: per-block ELBO loss, summed across blocks.
  - Enables KV caching for previously-generated blocks during inference.
- [ ] `src/diffusion_lm/diffusion/continuous.py` — `ContinuousDiffusionProcess`: `[M]`
  - `forward_process(x0_emb, t)`: `x_t = sqrt(alpha_bar) * x0_emb + sqrt(1 - alpha_bar) * noise`.
  - `compute_loss(predicted_x0, x0_emb, t)`: `F.mse_loss(predicted_x0, x0_emb)`.

**Tests — `tests/test_schedules.py`:**
- `test_alpha_boundary_conditions`: for all three schedules, `alpha(0) ≈ 1.0` and `alpha(1) ≈ 0.0`.
- `test_alpha_monotone_decreasing`: alpha strictly decreasing at `t=[0.1, 0.3, 0.5, 0.7, 0.9]`.
- `test_weight_positive`: `weight(t) > 0` everywhere in `(0, 1)`.

**Tests — `tests/test_diffusion.py`:**
- `test_masked_forward_masking_ratio`: at `t=0.5`, ~50% of tokens masked (±5% over large batch).
- `test_masked_compute_loss_shape`: loss is scalar, no NaN/Inf.
- `test_antithetic_timestep_coverage`: timesteps cover `[0, 1)` uniformly across batch.
- `test_sft_prompt_protection`: with `loss_mask` excluding prompt, gradient is zero on prompt tokens.
- `test_block_diffusion_structure`: block boundaries preserved, inter-block tokens not leaked.

**Gate:** `uv run pytest tests/test_schedules.py tests/test_diffusion.py -v`

---

## Phase 2: Model Architecture
**Goal:** Bidirectional transformer backbone, masked diffusion LM, continuous diffusion LM.

### 2a. BidirectionalTransformer backbone
- [ ] `src/diffusion_lm/models/backbone.py` — `BidirectionalTransformer(nn.Module)`: `[M]`
  - `__init__(config: ModelConfig)`: load via `from_pretrained` (default) or `from_config`.
  - **DO NOT use `_update_causal_mask` monkey-patch** — deprecated in Transformers v5.10, breaks `torch.compile`. See research.md for details.
  - `forward(input_ids, attention_mask=None, **kwargs)`:
    - Build explicit bidirectional mask: `bidi_mask = torch.zeros(B, 1, L, L, dtype=self.dtype, device=device)`.
    - If `attention_mask` is provided (e.g. block-diagonal for packed sequences), use it directly as the 4D mask.
    - Otherwise use the all-zeros 4D mask (full bidirectional).
    - Pass to `self.transformer(input_ids=input_ids, attention_mask=bidi_mask, ...)`.
    - Returns `outputs.logits` shape `(B, L, V)`.
  - For Flash Attention 2: `flash_attn_func(..., causal=False)` — pass explicitly, do not rely on config.
- [ ] `add_mask_token(model, tokenizer)`: `tokenizer.add_special_tokens({"mask_token": "[MASK]"})`, resize embeddings, return `mask_token_id`. `[S]`

### 2b. MaskedDiffusionLM
- [ ] `src/diffusion_lm/models/masked_diffusion_lm.py` — `MaskedDiffusionLM(nn.Module)`: `[M]`
  - Composes `BidirectionalTransformer` + `MaskedDiffusionProcess` (or `BlockDiffusionProcess`).
  - `forward(input_ids, attention_mask=None, labels=None, prompt_mask=None)`:
    1. Sample `t` via antithetic sampling.
    2. `corrupted, mask = diffusion.forward_process(input_ids, t)`.
    3. If `prompt_mask`: `corrupted = torch.where(prompt_mask, input_ids, corrupted)` — protect prompt BEFORE backbone.
    4. `logits = backbone(corrupted, attention_mask)`.
    5. `loss_mask = ~prompt_mask if prompt_mask else None`.
    6. `loss = diffusion.compute_loss(logits, input_ids, corrupted, t, loss_mask)`.
    7. Return `{"loss": loss, "logits": logits}`.

### 2c. ContinuousDiffusionLM
- [ ] `src/diffusion_lm/models/continuous_diffusion_lm.py` — `ContinuousDiffusionLM(nn.Module)`: `[L]`
  - Learned input projection, rounding head, lightweight timestep embedding `nn.Embedding(1000, d_model)`.
  - `round_to_tokens(embeddings)`: nearest-neighbor in embedding table via cosine similarity; straight-through estimator during training.
  - `forward(input_ids, t=None)`: embed → project → add noise → denoise → MSE loss + rounding CE loss.

**Tests — `tests/test_models.py`:**
- `test_bidirectional_attention`: change token at position 5; verify logits at position 3 change.
- `test_no_causal_mask_leakage`: verify the explicit 4D mask approach works across Transformers versions (check attention pattern, not implementation).
- `test_forward_pass_shapes`: `logits.shape == (B, L, V)` for both model types.
- `test_masked_diffusion_lm_loss_scalar`: `outputs["loss"].shape == ()`, not NaN.
- `test_prompt_mask_protection`: no gradient to prompt token embeddings when `prompt_mask` set.
- `test_mask_token_added`: after `add_mask_token()`, embeddings have correct shape.

**Gate:** `uv run pytest tests/test_models.py -v`

---

## Phase 3: Data Pipeline and Training
**Goal:** End-to-end training step works for pretrain, SFT, DPO, and GRPO.

### 3a. Data loaders
- [ ] `src/diffusion_lm/data/pretraining.py` — `PretrainingDataset`: `[M]`
  - `tokenize_and_group(dataset, tokenizer, block_size)`: concatenate, chunk, discard remainder.
  - `streaming=True` via HF `datasets`.
- [ ] `src/diffusion_lm/data/sft.py` — `SFTDataset`: `[M]`
  - Single-turn `{prompt, response}` and multi-turn `{messages: [{role, content}]}`.
  - Produces `(input_ids, prompt_mask)` where `prompt_mask[i] = True` at prompt positions.
- [ ] `src/diffusion_lm/data/preference.py` — `PreferenceDataset`: `[S]`
  - Produces `(chosen_input_ids, rejected_input_ids, prompt_mask)`.
- [ ] `src/diffusion_lm/data/collators.py` — three collators: `[M]`
  - `RandomTruncateCollator`: truncates with `random_truncation_ratio` probability. Emits per-sample block-diagonal attention mask for packed sequences (prevents cross-sample attention leakage).
  - `SFTCollator`: pads `input_ids` + `prompt_mask` together.
  - `DPOCollator`: pads both `chosen_input_ids` and `rejected_input_ids`.

### 3b. Trainers
- [ ] `src/diffusion_lm/trainers/base.py` — `DiffusionTrainer(HFTrainer)`: `[M]`
  - Override `compute_loss(model, inputs, return_outputs=False)`.
  - Add `NanLossCallback`: detect NaN loss, log diagnostic, stop training early.
- [ ] `src/diffusion_lm/trainers/pretraining.py` — `PretrainingTrainer`: pass `RandomTruncateCollator` as default. `[S]`
- [ ] `src/diffusion_lm/trainers/sft.py` — `SFTTrainer`: validate `prompt_mask` present, pass `SFTCollator`. `[S]`
- [ ] `src/diffusion_lm/trainers/dpo.py` — `DiffusionDPOTrainer`: `[L]`
  - `ref_model` with optional CPU offload.
  - `_estimate_log_prob(model, input_ids, prompt_mask, n_mc)`: shared antithetic `t` across policy + ref.
  - `compute_loss`: 4 ELBO estimates → DPO loss. Ref passes inside `torch.no_grad()`.
- [ ] `src/diffusion_lm/trainers/grpo.py` — `DiffusionGRPOTrainer` (diffu-GRPO, arXiv:2504.12216): `[L]`
  - `sample_completions(model, prompt_ids, K)`: generate K completions via `FirstHittingSampler`.
  - `_estimate_log_prob(model, prompt_ids, completion_ids, n_mc)`: ELBO averaging with antithetic t.
  - `compute_loss`: group-relative advantage from reward model scores → GRPO clipped objective.
  - Config: `group_size=8`, `clip_ratio=0.2`, `n_mc_samples=8`.

### 3c. Entry-point scripts
- [ ] `scripts/pretrain.py`: parse configs via `HfArgumentParser`, build dataset + `MaskedDiffusionLM`, run `PretrainingTrainer.train()`. `[M]`
- [ ] `scripts/sft.py`: same with `SFTDataset` + `SFTTrainer`. `[M]`
- [ ] `scripts/dpo.py`: load SFT checkpoint as policy and frozen ref, run `DiffusionDPOTrainer`. `[M]`
- [ ] `scripts/grpo.py`: load SFT checkpoint, reward model, run `DiffusionGRPOTrainer`. `[M]`

**Tests — `tests/test_trainers.py`:**
- `test_pretrain_one_step`: one optimizer step, random data, `gpt2` backbone, CPU.
- `test_sft_one_step`: with `prompt_mask`; loss only over response tokens.
- `test_dpo_one_step`: DPO trainer step with tiny frozen ref.
- `test_grpo_one_step`: GRPO trainer step with mock reward model.
- `test_loss_decreases`: over 10 steps on fixed batch, loss decreases.
- `test_nan_loss_callback`: inject NaN loss, verify callback triggers early stop.
- `test_packed_sequence_mask`: collator emits block-diagonal mask when packing > 1 sequence per row.

**Gate:**
```
uv run pytest tests/test_trainers.py -v && \
uv run python scripts/pretrain.py \
  --model_name_or_path gpt2 \
  --process_type masked \
  --schedule_type linear \
  --max_steps 10 \
  --per_device_train_batch_size 2 \
  --output_dir /tmp/smoke-test \
  --no_cuda
```

---

## Phase 4: Generation / Sampling
**Goal:** Trained models can generate coherent text. First-Hitting Sampler is the default.

### 4a. Base sampler and output type
- [ ] `src/diffusion_lm/samplers/base.py` — abstract `Sampler` with `generate(model, prompt_ids, config) -> SamplerOutput`; `SamplerOutput(sequences: Tensor, scores: Tensor | None)`. `[S]`

### 4b. First-Hitting Sampler (default)
- [ ] `src/diffusion_lm/samplers/first_hitting_sampler.py` — `FirstHittingSampler(Sampler)`: `[M]`
  - Initialize: `x = concat(prompt_ids, mask_token * gen_len)`.
  - Single forward pass → score function `s(x, t)` at masked positions.
  - Sample first-hitting time `τ_i ~ Exp(rate=score_i)` for each masked token independently.
  - Reveal tokens in order of ascending `τ_i`.
  - **Running Confidence Remasking** (enabled by default via `gen_config.running_confidence_remasking`):
    - After each reveal, check confidence of revealed token.
    - If confidence < dynamic threshold, allow re-masking (prevents Answer Backslide).
  - Suppress EOS logit at non-final positions.
  - Classifier-free guidance: `logits = uncond + (1 + guidance_scale) * (cond - uncond)`.

### 4c. Block Sampler (BD3LM)
- [ ] `src/diffusion_lm/samplers/block_sampler.py` — `BlockSampler(Sampler)`: `[M]`
  - AR over blocks: generate left-to-right one block at a time.
  - Within each block: `FirstHittingSampler` logic.
  - KV cache: save key/value tensors for completed blocks; only run attention over new block.
  - Enables long-sequence generation with sub-quadratic memory.

### 4d. Continuous Sampler
- [ ] `src/diffusion_lm/samplers/continuous_sampler.py` — `ContinuousSampler(Sampler)`: `[M]`
  - DDPM reverse: `x_{t-1} = sqrt(alpha_{t-1}) * predicted_x0 + sqrt(1 - alpha_{t-1}) * noise`.
  - Final step: `round_to_tokens(x_0_emb)` via exact nearest-neighbor (no STE at inference).

### 4e. Cached Sampler
- [ ] `src/diffusion_lm/samplers/cached_sampler.py` — `CachedSampler(FirstHittingSampler)`: `[M]`
  - Track positions predicted with same token for K consecutive steps — skip recompute.
  - Cache invalidation when any position changes.
  - ~2-3x speedup; lower priority given FirstHittingSampler's 20x baseline gain.

### 4f. Generation script
- [ ] `scripts/generate.py`: load checkpoint, build sampler from `GenerationConfig`, tokenize prompt, call `sampler.generate()`, decode and print. Support `--sampler first_hitting|block|continuous|cached`. `[S]`

**Tests — `tests/test_samplers.py`:**
- `test_first_hitting_output_length`: generated sequence has correct length.
- `test_prompt_preserved`: prompt portion matches input exactly.
- `test_no_mask_tokens_remain`: no `mask_token_id` in output after generation.
- `test_valid_token_ids`: all output IDs in `[0, vocab_size)`.
- `test_eos_suppression`: EOS does not appear at non-final denoising positions.
- `test_running_confidence_remasking`: with RCR enabled, final outputs have higher confidence scores than without (statistical test over N runs).
- `test_block_sampler_matches_first_hitting`: outputs statistically equivalent for short sequences.

**Gate:**
```
uv run pytest tests/test_samplers.py -v && \
uv run python scripts/generate.py \
  --model_path /tmp/smoke-test \
  --prompt "hello" \
  --max_new_tokens 32 \
  --sampler first_hitting
```

---

## Phase 5: Evaluation and Hardening
**Goal:** Benchmark evaluation works, observability complete, project deployable.

### 5a. Evaluation
- [ ] `src/diffusion_lm/evaluation/perplexity.py` — `ELBOPerplexity`: `[M]`
  - `compute(model, dataset, num_timestep_samples=32, batch_size=8) -> dict`.
  - `ppl_bound = exp(mean_nll_per_token)`. Returns `{"ppl_bound": float, "nll_per_token": float}`.
  - Note: use `FirstHittingSampler` for honest evaluation — categorical sampling inflates PPL vs. AR models.
- [ ] `src/diffusion_lm/evaluation/lm_eval_adapter.py` — `DiffusionLMEvalAdapter`: `[L]`
  - Implements `lm_eval.api.model.LM` interface (pin lm-eval version).
  - `loglikelihood(requests)`: ELBO with `n_samples=32`. Returns `(log_prob, is_greedy=False)`.
  - `generate_until(requests)`: `FirstHittingSampler.generate()`, truncate at stop sequences.
- [ ] `scripts/evaluate.py`: load checkpoint, run `ELBOPerplexity` and/or `DiffusionLMEvalAdapter`, write results JSON. `[M]`

### 5b. Error handling and observability
- [ ] Add structured logging via `loguru` throughout trainers, samplers, data loaders. No bare `print()`. `[S]`
- [ ] Retry with exponential backoff for HF Hub downloads in `data/pretraining.py`. `[S]`
- [ ] Validate `DiffusionConfig.mask_token_id != -1` before training; raise `ValueError` with clear message. `[S]`
- [ ] `NanLossCallback(TrainerCallback)`: detect NaN loss, log diagnostic, early stop. `[S]`
- [ ] Verify bidirectionality check utility: `assert_bidirectional(model, tokenizer)` — change position 5, check position 3 changes. Run in `DiffusionTrainer.__init__` in debug mode. `[S]`

### 5c. Deployment
- [ ] `Dockerfile`: multi-stage, `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` base. Install `flash-attn` in build stage. Final stage copies `src/` and `scripts/`. `[M]`
- [ ] `docker-compose.yml`: mounts `checkpoints/` and `.env`. `[S]`
- [ ] `README.md`: installation, quickstart, SFT example, generation example, benchmark example. `[M]`
- [ ] Fill in all YAML configs under `configs/`. `[S]`

**Gate:**
```
uv run pytest tests/ -v && \
uv run python scripts/evaluate.py \
  --model_path /tmp/smoke-test \
  --benchmarks perplexity \
  --dataset wikitext \
  --split test && \
docker build -t diffusion-lm .
```

---

## Phase Summary

| Phase | Focus | Tasks | Gate Command |
|-------|-------|-------|-------------|
| 0 | Scaffold | 9 | `ruff check . && pytest --co` |
| 1 | Math core (schedules + diffusion) | 10 | `pytest test_schedules.py test_diffusion.py -v` |
| 2 | Model architecture | 7 | `pytest test_models.py -v` |
| 3 | Data + trainers (pretrain/SFT/DPO/GRPO) | 12 | `pytest test_trainers.py -v && scripts/pretrain.py --max_steps 10` |
| 4 | Generation / samplers | 9 | `pytest test_samplers.py -v && scripts/generate.py` |
| 5 | Evaluation + hardening | 11 | `pytest tests/ -v && scripts/evaluate.py && docker build` |

**Total: 58 tasks across 6 phases.**
