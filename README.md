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

## Validated Models

The framework has been tested with the following backbones:

| Model | Params | Notes |
|-------|--------|-------|
| `Qwen/Qwen3-0.6B` | 0.6B | Recommended default. Pre-trained diffusion checkpoint available at `dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1` for benchmarking. Requires `transformers>=4.51`. |
| `Qwen/Qwen2.5-0.5B` | 0.5B | Same family as Dream-v0 and Tiny-A2D (validated diffusion paths). Apache 2.0. |
| `meta-llama/Llama-3.2-1B` | 1B | Same family as DiffuLLaMA (ICLR 2025). GQA works correctly with 4D mask. Requires HF token. |
| `gpt2` | 124M | Fastest for smoke tests. Pre-validated via DiffuGPT (ICLR 2025). |

**Avoid** Mistral and Phi-3-mini — both use sliding window attention that operates below the 4D mask injection level.

## Installation

```bash
git clone https://github.com/sjonas50/diffusion.git
cd diffusion
uv sync
```

Requires Python 3.11+. For Flash Attention 2 (recommended for GPU training):

```bash
pip install flash-attn
```

## Quickstart

### 1. Pretrain / AR-to-dLLM adaptation

Fine-tune Qwen3-0.6B on financial news with the masked diffusion objective:

```bash
uv run python scripts/pretrain.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
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
  --prompt "The Federal Reserve announced" \
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

## GPU Training (RunPod / Cloud)

For GPU training on RunPod or similar cloud providers, use Flash Attention 2 for best performance:

```bash
python3 scripts/pretrain.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --process_type masked --schedule_type linear \
  --block_size 512 --dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --dataset_name ashraq/financial-news-articles \
  --text_column text --dataset_split train \
  --output_dir ./checkpoints/finance-qwen3 \
  --max_steps 2000 --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 --bf16 true \
  --learning_rate 3e-5 --lr_scheduler_type cosine \
  --warmup_steps 200 --weight_decay 0.1 \
  --max_grad_norm 1.0 --logging_steps 25 \
  --save_steps 500 --gradient_checkpointing true
```

**Attention backends:** `--attn_implementation` supports `eager` (CPU/MPS), `sdpa` (GPU, no flash-attn needed), and `flash_attention_2` (fastest, requires `pip install flash-attn`). FA2 enables bidirectional attention via `is_causal=False` on all attention modules.

See [`docs/runpod-setup.md`](docs/runpod-setup.md) for detailed RunPod deployment guide including Docker image selection, SSH setup, and cost optimization.

## Finance Domain Example

The framework has been tested end-to-end on `ashraq/financial-news-articles` (306k Reuters/Bloomberg articles) with Qwen3-0.6B:

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from diffusion_lm.data.pretraining import tokenize_and_group

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
raw_ds = load_dataset("ashraq/financial-news-articles", split="train")
chunks = list(tokenize_and_group(raw_ds, tokenizer, block_size=512, text_column="text"))
```

Other recommended finance datasets:
- `PleIAs/SEC` — 373k SEC 10-K annual reports, CC0 license (use streaming for full dataset)
- `sujet-ai/Sujet-Finance-Instruct-177k` — 177k QA pairs for SFT, Apache 2.0

## Testing

```bash
# Fast unit tests (54 tests, ~4s)
uv run pytest tests/ -m "not slow"

# Integration tests against real model weights (downloads ~1-2GB per model)
uv run pytest tests/test_model_compat.py -v -m slow
```

The integration tests verify bidirectionality, forward pass sanity, generation, and mask token handling for each supported model.

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
configs/models/      # qwen3-0-6b.yaml, qwen2-5-0-5b.yaml, llama-3-2-1b.yaml, small.yaml, ...
tests/               # 54 unit tests + 12 slow integration tests across all supported models
```

## Key Design Decisions

**Explicit 4D attention mask** — rather than monkey-patching `_update_causal_mask` (removed in Transformers 5.10), we inject an all-zeros float mask of shape `(B, 1, L, L)` directly in `forward()`. This is `torch.compile`-safe and works across all HF model architectures.

**Antithetic timestep sampling** — draw one `u ~ U[0,1)`, set `t_i = (u + i/B) mod 1`. Reduces gradient variance ~B-fold at zero cost.

**ELBO-weighted loss** — divide CE by `p_mask(t)` at each position. Mandatory for correct ELBO estimation; omitting it biases training.

**First-Hitting Sampler as default** — standard categorical sampling exploits a mathematical inaccuracy that inflates benchmarks. The First-Hitting Sampler is theoretically correct and ~20x faster.

**Tied weight handling** — GPT-2, Qwen, and LLaMA all share `embed_tokens.weight` with `lm_head.weight`. `DiffusionTrainer._save()` temporarily breaks this tie before writing safetensors checkpoints and restores it immediately after.

## References

- [LLaDA](https://arxiv.org/abs/2502.09992) — Large Language Diffusion with mAsking
- [Mercury 2](https://arxiv.org/abs/2506.17298) — Inception Labs, 1000+ tok/s on H100
- [BD3LM](https://arxiv.org/abs/2503.09573) — Block Diffusion, ICLR 2025 Oral
- [diffu-GRPO](https://arxiv.org/abs/2504.12216) — First RL pipeline for dLLMs
- [First-Hitting Sampler](https://arxiv.org/abs/2409.02908) — Correct and fast sampling
- [MDLM](https://arxiv.org/abs/2406.07524) — Masked Diffusion Language Models (NeurIPS 2024)
- [DiffuLLaMA](https://arxiv.org/abs/2410.17891) — LLaMA-family AR-to-dLLM adaptation (ICLR 2025)
