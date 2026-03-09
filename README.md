# diffusion-lm

A training framework for diffusion-based language models (dLLMs) supporting masked diffusion (LLaDA/Mercury style), block diffusion (BD3LM), and continuous embedding diffusion.

Wraps any HuggingFace causal LM backbone with bidirectional attention and adds a full training and inference pipeline. Supports AR-to-dLLM adaptation, SFT, DPO, GRPO, and generation via First-Hitting Sampler.

## Features

- **AR-to-dLLM adaptation** — start from any HF causal LM checkpoint (~500x less compute than training from scratch)
- **Masked diffusion** (LLaDA/Mercury style) with ELBO-weighted cross-entropy loss and antithetic timestep sampling
- **Block diffusion** (BD3LM, ICLR 2025 Oral) — AR over blocks, masked within; enables KV caching and 13% PPL improvement
- **First-Hitting Sampler** — theoretically correct, 20x faster than naive categorical sampling; default for all generation
- **Running Confidence Remasking** — prevents Answer Backslide (9.8% MATH-500 failure rate without it)
- **GRPO alignment** (diffu-GRPO, arXiv:2504.12216) — first working RL pipeline for dLLMs
- **DPO alignment** — ELBO-based log-likelihood estimation with shared antithetic timesteps
- **Explicit 4D attention mask** — compatible with Transformers 5.10+, `torch.compile`-safe
- **HuggingFace Trainer extension** — inherits distributed training (DDP/FSDP/DeepSpeed), checkpointing, W&B logging

## Installation

```bash
git clone https://github.com/sjonas50/diffusion.git
cd diffusion
uv sync
```

Requires Python 3.11+. For Flash Attention 2 (recommended for training):

```bash
pip install flash-attn --no-build-isolation
```

## Quickstart

### 1. Pretrain / AR-to-dLLM adaptation

Fine-tune a GPT-2 checkpoint with the masked diffusion objective:

```bash
uv run python scripts/pretrain.py \
  --model_name_or_path gpt2 \
  --process_type masked \
  --schedule_type linear \
  --max_steps 100000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --output_dir ./checkpoints/pretrain
```

### 2. Supervised Fine-Tuning (SFT)

```bash
uv run python scripts/sft.py \
  --model_path ./checkpoints/pretrain \
  --dataset_name your_dataset \
  --max_steps 5000 \
  --output_dir ./checkpoints/sft
```

### 3. GRPO Alignment (recommended for reasoning)

```bash
uv run python scripts/grpo.py \
  --model_path ./checkpoints/sft \
  --max_steps 500 \
  --group_size 8 \
  --output_dir ./checkpoints/grpo
```

### 4. Generate

```bash
uv run python scripts/generate.py \
  --model_path ./checkpoints/grpo \
  --prompt "The meaning of life is" \
  --sampler first_hitting \
  --num_steps 64 \
  --max_new_tokens 128
```

Available samplers: `first_hitting` (default), `block`, `continuous`, `cached`.

### 5. Evaluate

```bash
uv run python scripts/evaluate.py \
  --model_path ./checkpoints/grpo \
  --benchmarks perplexity \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split test
```

## Docker

```bash
# Build
docker build -t diffusion-lm .

# Train with GPU
docker compose run train

# Generate
docker compose run generate
```

## Project Structure

```
src/diffusion_lm/
├── config/          # ModelConfig, DiffusionConfig, TrainingArguments, GenerationConfig
├── schedules/       # Linear, cosine, log-linear noise schedules
├── diffusion/       # MaskedDiffusionProcess, BlockDiffusionProcess, ContinuousDiffusionProcess
├── models/          # BidirectionalTransformer backbone, MaskedDiffusionLM, ContinuousDiffusionLM
├── data/            # Collators (RandomTruncate, SFT, DPO), PretrainingDataset, SFTDataset
├── trainers/        # DiffusionTrainer, SFTTrainer, DiffusionDPOTrainer, DiffusionGRPOTrainer
├── samplers/        # FirstHittingSampler, BlockSampler, ContinuousSampler, CachedSampler
└── evaluation/      # ELBOPerplexity, DiffusionLMEvalAdapter
scripts/             # pretrain.py, sft.py, dpo.py, grpo.py, generate.py, evaluate.py
configs/             # YAML configs for models, diffusion processes, training runs
tests/               # 54 tests across schedules, diffusion, models, trainers, samplers
```

## Key Design Decisions

**Explicit 4D attention mask** — rather than monkey-patching `_update_causal_mask` (removed in Transformers 5.10), we inject an all-zeros float mask of shape `(B, 1, L, L)` directly in `forward()`. This is `torch.compile`-safe and works across all HF model architectures.

**Antithetic timestep sampling** — draw one `u ~ U[0,1)`, set `t_i = (u + i/B) mod 1`. Reduces gradient variance ~B-fold at zero cost.

**ELBO-weighted loss** — divide CE by `p_mask(t)` at each position. Mandatory for correct ELBO estimation; omitting it biases training.

**First-Hitting Sampler as default** — standard categorical sampling exploits a mathematical inaccuracy that inflates benchmarks. The First-Hitting Sampler is theoretically correct and ~20x faster.

## References

- [LLaDA](https://arxiv.org/abs/2502.09992) — Large Language Diffusion with mAsking
- [Mercury 2](https://arxiv.org/abs/2506.17298) — Inception Labs, 1000+ tok/s on H100
- [BD3LM](https://arxiv.org/abs/2503.09573) — Block Diffusion, ICLR 2025 Oral
- [diffu-GRPO](https://arxiv.org/abs/2504.12216) — First RL pipeline for dLLMs
- [First-Hitting Sampler](https://arxiv.org/abs/2409.02908) — Correct and fast sampling
- [MDLM](https://arxiv.org/abs/2406.07524) — Masked Diffusion Language Models (NeurIPS 2024)
