# Diffusion LLM Training Framework

## Context

**What are Diffusion LLMs?** A new paradigm for text generation that replaces autoregressive next-token prediction with a diffusion process. Instead of generating one token at a time left-to-right, diffusion LLMs start with a fully corrupted sequence and iteratively refine it, generating multiple tokens in parallel. The model backbone is still a standard Transformer, but uses **bidirectional attention** (no causal mask).

**Why build this?** Mercury 2 (by Inception Labs, Feb 24, 2026) demonstrated that diffusion LLMs can achieve 727–1,196 tokens/sec — 13x faster than Claude 4.5 Haiku — while matching its quality tier (AIME 2025: 91.1, GPQA Diamond: 73.6). Note: TTFT is 3.48s (all diffusion steps must complete before first token streams). LLaDA (Feb 2025) proved the approach scales to 8B parameters. **As of 2026, the dominant paradigm is AR-to-dLLM adaptation** — fine-tuning a pretrained autoregressive checkpoint with ~1B tokens of diffusion objective training (~500x less compute than from scratch). This framework supports both adaptation and from-scratch training.

**Two diffusion approaches to implement:**

1. **Masked Diffusion** (LLaDA/Mercury style): Forward process randomly masks tokens with `[MASK]`; reverse process iteratively unmasks. Training objective = cross-entropy on masked positions, weighted by masking probability. Simpler, proven at scale.

2. **Continuous Embedding Diffusion** (Diffusion-LM style): Forward process adds Gaussian noise to token embeddings; reverse process denoises. Requires a rounding step to convert continuous embeddings back to discrete tokens. More expressive but less proven.

**Key technical insights:**
- The Transformer backbone is unchanged — only the attention mask switches from causal to bidirectional
- Masked diffusion loss = `CE(logits[masked], targets[masked]) / p_mask(t)`, equivalent to ELBO upper bound
- Timestep `t` does NOT need to be input to the model (proven by RADD paper)
- SFT works by only masking response tokens (prompt stays clean)
- DPO adapts by replacing autoregressive log-probs with ELBO estimates
- Flash Attention 2 enables bidirectional attention by passing a non-causal attention mask

**References:** [Mercury 1 paper](https://arxiv.org/abs/2506.17298), [LLaDA paper](https://arxiv.org/abs/2502.09992), [MDLM (NeurIPS 2024)](https://github.com/kuleshov-group/mdlm), [dLLM library](https://github.com/ZHZisZZ/dllm), [BD3LM](https://arxiv.org/abs/2503.09573), [First-Hitting Sampler](https://arxiv.org/abs/2409.02908), [diffu-GRPO/d1](https://arxiv.org/abs/2504.12216), [CDLM distillation](https://arxiv.org/pdf/2511.19269), [Dream 7B](https://arxiv.org/html/2508.15487v1)

---

## Project Structure

```
diffusion/
├── pyproject.toml
├── diffusion_lm/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── model.py              # ModelConfig dataclass
│   │   ├── diffusion.py          # DiffusionConfig (process type, schedule, params)
│   │   ├── training.py           # DiffusionTrainingArguments (extends HF TrainingArguments)
│   │   └── generation.py         # GenerationConfig (sampler, steps, temperature)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py           # BidirectionalTransformer (wraps any HF causal LM)
│   │   ├── masked_diffusion_lm.py    # MaskedDiffusionLM
│   │   └── continuous_diffusion_lm.py # ContinuousDiffusionLM
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract DiffusionProcess
│   │   ├── masked.py             # MaskedDiffusionProcess (forward mask + loss)
│   │   ├── block.py              # BlockDiffusionProcess (BD3LM: AR over blocks, masked within)
│   │   └── continuous.py         # ContinuousDiffusionProcess (Gaussian noise + loss)
│   ├── schedules/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract NoiseSchedule: alpha(t), weight(t)
│   │   ├── linear.py             # alpha(t) = 1 - t
│   │   ├── cosine.py             # alpha(t) = cos(pi/2 * t)
│   │   └── loglinear.py          # alpha(t) = exp(-sigma(t))
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract Sampler + SamplerOutput
│   │   ├── first_hitting_sampler.py   # First-Hitting Sampler (default, 20x faster, theoretically correct)
│   │   ├── block_sampler.py           # BD3LM block diffusion sampler (KV cache)
│   │   ├── continuous_sampler.py      # DDPM denoising + rounding
│   │   └── cached_sampler.py         # Cached predictions for speedup
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base.py               # DiffusionTrainer (extends HF Trainer)
│   │   ├── pretraining.py        # PretrainingTrainer
│   │   ├── sft.py                # SFTTrainer (prompt-aware masking)
│   │   ├── dpo.py                # DiffusionDPOTrainer (ELBO-based)
│   │   └── grpo.py               # DiffusionGRPOTrainer (diffu-GRPO, arXiv:2504.12216)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pretraining.py        # Streaming tokenize-and-group
│   │   ├── sft.py                # SFT with prompt_mask boundaries
│   │   ├── preference.py         # Chosen/rejected pairs for DPO
│   │   └── collators.py          # Padding + random truncation collators
│   └── evaluation/
│       ├── __init__.py
│       ├── perplexity.py         # ELBO-based perplexity
│       └── lm_eval_adapter.py    # lm-evaluation-harness integration
├── scripts/
│   ├── pretrain.py
│   ├── sft.py
│   ├── dpo.py
│   ├── generate.py
│   └── evaluate.py
├── configs/
│   ├── models/
│   │   ├── small.yaml            # ~125M (GPT-2 scale)
│   │   ├── medium.yaml           # ~350M
│   │   ├── large.yaml            # ~1.3B
│   │   └── 8b.yaml               # ~8B (LLaMA-3 scale)
│   ├── diffusion/
│   │   ├── masked_linear.yaml
│   │   ├── masked_cosine.yaml
│   │   └── continuous_cosine.yaml
│   └── training/
│       ├── pretrain.yaml
│       ├── sft.yaml
│       └── dpo.yaml
└── tests/
    ├── test_diffusion.py
    ├── test_schedules.py
    ├── test_models.py
    ├── test_samplers.py
    └── test_trainers.py
```

---

## Implementation Plan

### Phase 1: Foundation — Configs, Schedules, Diffusion Processes

**Goal:** Build the mathematical core that everything else depends on.

#### 1a. Configuration system (`config/`)
- `ModelConfig`: `model_name_or_path`, `init_from_pretrained`, `attn_implementation="flash_attention_2"`, `dtype`, LoRA params
- `DiffusionConfig`: `process_type` ("masked"/"continuous"), `schedule_type`, `mask_token_id`, `time_epsilon=1e-3`
- `DiffusionTrainingArguments`: extends HF `TrainingArguments` with `random_truncation_ratio`, `dpo_beta`, `dpo_num_mc_samples`
- `GenerationConfig`: `num_steps`, `temperature`, `block_size`, `remasking` ("random"/"low_confidence"), `max_new_tokens`

#### 1b. Noise schedules (`schedules/`)
Abstract base: `alpha(t) -> keep probability`, `alpha_derivative(t)`, `weight(t) = -alpha'(t)/(1-alpha(t))`
- `LinearSchedule`: `alpha(t) = 1 - t` (used by LLaDA)
- `CosineSchedule`: `alpha(t) = cos(pi/2 * t)` (used by MDLM)
- `LogLinearSchedule`: `alpha(t) = exp(-(-log(1-(1-eps)*t)))` (used by MDLM)

#### 1c. Diffusion processes (`diffusion/`)

**MaskedDiffusionProcess:**
```python
def forward_process(self, x0, t):
    p_mask = 1 - self.schedule.alpha(t)   # per-sample masking probability
    random_mask = torch.rand_like(x0.float()) < p_mask  # independent per-token
    corrupted = torch.where(random_mask, self.mask_token_id, x0)
    return corrupted, random_mask

def compute_loss(self, logits, x0, corrupted, t, loss_mask=None):
    masked = (corrupted == self.mask_token_id)
    if loss_mask is not None:
        masked = masked & loss_mask
    ce = F.cross_entropy(logits.view(-1, V), x0.view(-1), reduction='none').view(x0.shape)
    p_mask = self.schedule.mask_probability(t.unsqueeze(-1))
    weighted = ce / p_mask.clamp(min=1e-5)  # ELBO-unbiased weighting
    return (weighted * masked).sum() / masked.sum().clamp(min=1)
```

**ContinuousDiffusionProcess:**
```python
def forward_process(self, x0_emb, t):
    alpha_bar = self.schedule.alpha_bar(t).unsqueeze(-1).unsqueeze(-1)
    noise = torch.randn_like(x0_emb)
    x_t = torch.sqrt(alpha_bar) * x0_emb + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise

def compute_loss(self, predicted_x0, x0_emb, t):
    return F.mse_loss(predicted_x0, x0_emb)
```

### Phase 2: Model Architecture

**Goal:** Wrap HF transformers for bidirectional diffusion.

#### 2a. BidirectionalTransformer (`models/backbone.py`)
- Takes any `AutoModelForCausalLM` (LLaMA, Mistral, Qwen, GPT-2)
- **DO NOT use `_update_causal_mask` monkey-patch** — deprecated, removed in Transformers v5.10, and breaks `torch.compile`. Silent causal fallback is a real failure mode (Gemma3 bug #39389).
- **Safe approach:** Subclass model `forward()` and inject an explicit all-zeros 4D float mask `(batch, 1, seq_len, seq_len)`. Bypasses all HF internal mask generation machinery.
- For Flash Attention 2: pass `is_causal=False` explicitly to `flash_attn_func` — do not rely on config propagation.
- Also supports AR-to-dLLM adaptation (`init_from_pretrained=True`) as primary recommended path.
- Returns logits `(B, L, V)` from bidirectional attention

```python
class BidirectionalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.init_from_pretrained:
            self.transformer = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.dtype),
                attn_implementation=config.attn_implementation,
            )
        else:
            hf_config = AutoConfig.from_pretrained(config.model_name_or_path)
            self.transformer = AutoModelForCausalLM.from_config(hf_config, ...)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        # Inject explicit bidirectional mask — bypasses HF's internal causal mask logic.
        # All-zeros float mask = no masking = full bidirectional attention.
        # This approach is stable across Transformers versions and works with torch.compile.
        bidi_mask = torch.zeros(B, 1, L, L, dtype=self.transformer.dtype, device=input_ids.device)
        outputs = self.transformer(input_ids=input_ids, attention_mask=bidi_mask, **kwargs)
        return outputs.logits
```

#### 2b. MaskedDiffusionLM (`models/masked_diffusion_lm.py`)
```python
def forward(self, input_ids, attention_mask=None, labels=None, prompt_mask=None):
    t = self.diffusion.sample_timesteps(B, device)
    corrupted, mask = self.diffusion.forward_process(input_ids, t)
    if prompt_mask is not None:
        corrupted = torch.where(prompt_mask, input_ids, corrupted)  # protect prompt
    logits = self.backbone(corrupted, attention_mask)
    loss_mask = ~prompt_mask if prompt_mask is not None else None
    loss = self.diffusion.compute_loss(logits, input_ids, corrupted, t, loss_mask)
    return {"loss": loss, "logits": logits}
```

#### 2c. ContinuousDiffusionLM (`models/continuous_diffusion_lm.py`)
- Adds learned input projection, rounding head, and timestep embedding
- Forward: embed tokens -> add noise -> denoise -> compute MSE + rounding CE loss
- `round_to_tokens(embeddings)`: nearest-neighbor lookup via `cdist` against embedding table

**Mask token handling:** Add `[MASK]` to tokenizer via `add_special_tokens`, resize model embeddings. The new embedding trains from scratch.

### Phase 3: Data Pipeline & Training

#### 3a. Pretraining data (`data/pretraining.py`)
- HF `datasets` with streaming
- `tokenize_and_group`: concatenate all texts, split into fixed-length chunks
- `RandomTruncateCollator`: with probability 1%, truncate to random length (LLaDA technique)

#### 3b. SFT data (`data/sft.py`)
- Support single-turn `{prompt, response}` and multi-turn `{messages: [{role, content}]}`
- Produces `prompt_mask: (B, L)` — True for prompt positions, False for response positions
- In multi-turn: all user turns = prompt, all assistant turns = response

#### 3c. Preference data (`data/preference.py`)
- Format: `{prompt, chosen, rejected}`
- Produces `chosen_input_ids`, `rejected_input_ids` with corresponding `prompt_mask`s

#### 3d. Trainers (`trainers/`)

**DiffusionTrainer (base):** extends HF `Trainer.compute_loss()` to apply diffusion forward process before computing loss. Gets automatic gradient accumulation, mixed precision, logging, checkpointing, and distributed training (DDP/FSDP/DeepSpeed).

**PretrainingTrainer:** adds random truncation. All tokens participate in diffusion.

**SFTTrainer:** ensures `prompt_mask` is present; only response tokens are masked/contribute to loss.

**DiffusionDPOTrainer:**
- Loads frozen reference model (copy of SFT checkpoint)
- Estimates `log p(response|prompt)` via Monte Carlo over timesteps (average diffusion loss over N sampled timesteps)
- DPO loss: `-log sigmoid(beta * ((logp_policy_chosen - logp_ref_chosen) - (logp_policy_rejected - logp_ref_rejected)))`
- Uses antithetic sampling: shared timesteps and masks between policy and reference to reduce variance
- Config: `n_mc_samples=8`, `beta=0.1`
- **Note: For reasoning tasks, prefer `DiffusionGRPOTrainer` (diffu-GRPO) — shown to match AR reasoning models on GSM8K/MATH500**

**DiffusionGRPOTrainer (`trainers/grpo.py`) — diffu-GRPO (arXiv:2504.12216):**
- First working RL pipeline for dLLMs. Adapts GRPO for parallel token generation.
- Group relative policy optimization: sample K completions per prompt, score with reward model, optimize relative advantage
- Key adaptation: log-prob estimation via ELBO averaging over sampled timesteps (same as DPO MC estimation)
- Config: `n_mc_samples=8`, `group_size=8`, `clip_ratio=0.2`
- Reference: [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1)

### Phase 4: Generation / Sampling

#### 4a. First-Hitting Sampler (`samplers/first_hitting_sampler.py`) — **Default sampler**
Replaces the standard confidence-based categorical sampler. Proven in arXiv:2409.02908 to be theoretically correct (standard categorical sampling exploits a mathematical inaccuracy that inflates benchmark scores vs. AR models). Drop-in replacement with **20x faster inference**.

Key properties:
- Each masked token is independently assigned a "first-hitting time" based on the score function
- Tokens are revealed in order of hitting times — no iterative remask/unmask loop
- Supports Running Confidence Remasking as an option to prevent "Answer Backslide" (9.8% MATH-500 failure rate where correct intermediate tokens get overwritten)
- Supports classifier-free guidance: `logits = uncond + (1 + scale) * (cond - uncond)`

```
1. Initialize: prompt_ids + [MASK]*gen_len
2. Forward pass → score function for all masked positions
3. Sample first-hitting times per token from score-parameterized distribution
4. Reveal tokens in hitting-time order (optionally: apply Running Confidence Remasking)
5. Return generated sequence
```

#### 4b. Block Diffusion Sampler (`samplers/block_sampler.py`) — **BD3LM variant**
For models trained with Block Diffusion (BD3LM, arXiv:2503.09573). Autoregressive over blocks with First-Hitting Sampler within each block. Enables KV caching for previously-generated blocks → significant speedup for long sequences.

#### 4c. Continuous sampler (`samplers/continuous_sampler.py`)
DDPM-style: `x_{t-1} = sqrt(alpha_{t-1}) * predicted_x0 + sqrt(1-alpha_{t-1}) * noise`
Final step: round to nearest tokens via embedding table lookup.

#### 4d. Cached sampler (`samplers/cached_sampler.py`)
Caches predictions for positions that stabilized across steps. ~2-3x speedup. Lower priority given First-Hitting Sampler's 20x gain.

### Phase 5: Evaluation

#### 5a. ELBO Perplexity (`evaluation/perplexity.py`)
- Sample N timesteps per sequence, average the diffusion loss
- `PPL_bound = exp(-ELBO_per_token)` — upper bound on true perplexity

#### 5b. lm-evaluation-harness adapter (`evaluation/lm_eval_adapter.py`)
- Implements `lm_eval.api.model.LM` interface
- `loglikelihood(context, continuation)`: estimate via ELBO with N=32 timestep samples
- `generate_until(context, stop)`: use diffusion sampler, truncate at stop sequences
- `is_greedy` always False (no greedy decoding notion in diffusion)

### Phase 6: Tests

- `test_schedules.py`: verify alpha(0)=1, alpha(1)=0, monotonicity, weight positivity
- `test_diffusion.py`: forward process produces correct masking ratio, loss computes without error
- `test_models.py`: bidirectional attention works (output differs from causal), forward pass shapes correct
- `test_samplers.py`: generation produces valid token IDs, respects prompt, correct lengths
- `test_trainers.py`: one training step runs without error, loss decreases over 10 steps

---

## Critical Implementation Details

1. **Disabling causal mask (UPDATED — DO NOT use monkey-patch)**: `_update_causal_mask` is deprecated and removed in Transformers v5.10. The new `masking_utils.create_causal_mask` path breaks `torch.compile(fullgraph=True)`. Silent causal fallback is a real failure mode (Gemma3 issue #39389 — model trains on wrong attention, no crash signal).
   **Correct approach:** Subclass model's `forward()` and inject explicit all-zeros 4D float mask `(batch, 1, seq_len, seq_len)`. For FA2: pass `is_causal=False` explicitly to `flash_attn_func`.
   Verify bidirectionality: changing token at position 5 must affect logits at position 3 (impossible with causal mask).

2. **Loss normalization**: Divide CE loss by `p_mask(t)` per-token for unbiased ELBO. For SFT, normalize by number of response tokens instead.

3. **Antithetic timestep sampling**: For a batch of B, sample one `u ~ U[0,1)`, create `t_i = (u + i/B) mod 1` scaled to `[eps, 1)`. Ensures uniform coverage, reduces gradient variance.

4. **Continuous diffusion rounding**: Use straight-through estimator during training (round in forward, pass gradients through as-if unrounded). Nearest-neighbor in embedding table at inference.

5. **EOS handling during generation**: Suppress EOS logits during intermediate steps, only allow in final step.

6. **DPO/GRPO memory**: Requires ~32 forward passes per step (4 ELBO estimates x 8 MC samples). Use gradient checkpointing, bf16, and CPU offloading for reference model.

7. **Cross-sample attention leakage in packed sequences**: When packing multiple sequences into one batch row, use per-sample block-diagonal attention masks in the data collator. Without this, tokens from different samples can attend to each other — silent quality degradation with no crash signal.

8. **"Answer Backslide" prevention**: 9.8% of MATH-500 failures have correct intermediate tokens overwritten in later steps. Enable Running Confidence Remasking in the sampler by default — free, no retraining required.

9. **Gradient variance**: Random masking causes ~14x higher variance than full-mask training. Consider MIRROR (anti-correlated mask pairs) for stabilizing training at scale.

---

## Verification Plan

### Smoke test (Phase 2 complete)
```bash
# Train a tiny masked diffusion model on a small dataset
python scripts/pretrain.py \
  --model_name_or_path gpt2 \
  --process_type masked \
  --schedule_type linear \
  --max_steps 100 \
  --per_device_train_batch_size 4 \
  --output_dir ./checkpoints/smoke-test
```
- Verify: loss decreases, no NaN, bidirectional attention confirmed

### Generation test (Phase 4 complete)
```bash
python scripts/generate.py \
  --model_path ./checkpoints/smoke-test \
  --prompt "The meaning of life is" \
  --num_steps 64 \
  --max_new_tokens 128
```
- Verify: produces coherent tokens (quality won't be great at this scale)

### SFT test (Phase 3 complete)
```bash
python scripts/sft.py \
  --model_path ./checkpoints/smoke-test \
  --dataset tatsu-lab/alpaca \
  --max_steps 50
```
- Verify: loss only computed on response tokens, prompt tokens unchanged

### Evaluation test (Phase 5 complete)
```bash
python scripts/evaluate.py \
  --model_path ./checkpoints/sft \
  --benchmarks perplexity \
  --dataset wikitext
```

### Unit tests
```bash
pytest tests/ -v
```
