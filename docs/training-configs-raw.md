# Raw Training Configs Extracted from dLLM Repositories

Last updated: 2026-03-09

This file contains verbatim config files and training scripts extracted from major dLLM
repositories. For synthesized recommendations, see `training-recipes.md`.

---

## 1. MDLM (kuleshov-group/mdlm) — master branch

### configs/config.yaml (FULL)

```yaml
defaults:
  - _self_
  - /callbacks: [checkpoint_every_n_steps, checkpoint_monitor, learning_rate_monitor]
  - /data: openwebtext
  - /model: small
  - /strategy: ddp
  - /noise: loglinear
  - /lr_scheduler: constant_warmup

mode: train  # train / ppl_eval / sample_eval
diffusion: absorbing_state
backbone: dit  # dit / dimamba / ar
parameterization: subs  # subs / d3pm / sedd
time_conditioning: False
T: 0  # 0 (continuous time) / 1000
subs_masking: False

seed: 1

loader:
  global_batch_size: 512
  eval_global_batch_size: ${.global_batch_size}
  batch_size: ${div_up:${.global_batch_size}, ${eval:${trainer.devices} * ${trainer.num_nodes}}}
  eval_batch_size: ${div_up:${.eval_global_batch_size}, ${eval:${trainer.devices} * ${trainer.num_nodes}}}
  num_workers: ${eval:"len(__import__('os').sched_getaffinity(0))"}
  pin_memory: True

sampling:
  predictor: ddpm_cache  # analytic, ddpm, ddpm_cache
  steps: 128
  noise_removal: True
  num_sample_batches: 2
  num_sample_log: 2
  semi_ar: False
  stride_length: 1
  num_strides: 1

training:
  ema: 0.9999
  antithetic_sampling: True
  importance_sampling: False
  sampling_eps: 1e-3
  change_of_variables: False

eval:
  checkpoint_path: ''
  disable_ema: False
  compute_generative_perplexity: False
  perplexity_batch_size: 8
  compute_perplexity_on_sanity: False
  gen_ppl_eval_model_name_or_path: gpt2-large
  generate_samples: True

optim:
  weight_decay: 0
  lr: 3e-4
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8

trainer:
  _target_: lightning.Trainer
  accelerator: cuda
  num_nodes: 1
  devices: ${device_count:}
  accumulate_grad_batches: ${div_up:${loader.global_batch_size}, ${eval:${trainer.devices} * ${loader.batch_size} * ${trainer.num_nodes}}}
  gradient_clip_val: 1.0
  precision: 'bf16'
  num_sanity_val_steps: 2
  max_steps: 1_000_000
  log_every_n_steps: 10
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  val_check_interval: 10000

wandb:
  project: text-diffusion

checkpointing:
  save_dir: ${cwd:}
  resume_from_ckpt: true
  resume_ckpt_path: ${.save_dir}/checkpoints/last.ckpt
```

### configs/model/small.yaml

```yaml
type: ddit
hidden_size: 768
n_blocks: 12
n_heads: 12
cond_dim: 128
length: 1024
scale_by_sigma: True
dropout: 0.1
tie_word_embeddings: False
```

### configs/lr_scheduler/constant_warmup.yaml

```yaml
_target_: transformers.get_constant_schedule_with_warmup
num_warmup_steps: 2500
```

### Key extracted values

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 3e-4 |
| beta1 | 0.9 |
| beta2 | 0.999 |
| eps | 1e-8 |
| Weight decay | 0 |
| EMA decay | 0.9999 |
| Global batch size | 512 |
| Gradient clip | 1.0 |
| Max steps | 1,000,000 |
| Precision | bf16 |
| Warmup steps | 2500 |
| LR schedule | Constant with warmup |
| Antithetic sampling | True |
| Sampling eps | 1e-3 |
| Seq length | 1024 |
| Val check interval | 10,000 steps |
| Framework | PyTorch Lightning + Hydra |

---

## 2. BD3LM (kuleshov-group/bd3lms) — main branch

BD3LM inherits the MDLM codebase and config structure. Config is nearly identical.

### configs/config.yaml (key differences from MDLM)

```yaml
# Same structure as MDLM, with these differences:
training:
  ema: 0.9999
  antithetic_sampling: True
  nll_eval: True  # BD3LM addition

optim:
  weight_decay: 0
  lr: 3e-4

trainer:
  max_steps: 1_000_000
  gradient_clip_val: 1.0
  precision: 'bf16'

wandb:
  project: BD3-LMs
```

### Training details from paper (arXiv 2503.09573)

- BD3-LMs are finetuned from an MDLM checkpoint trained for 850K steps on OpenWebText
- Additional training: 1M steps with block sizes 4, 8, 16
- Same optimizer config as MDLM (AdamW, lr=3e-4, weight_decay=0)
- Global batch size: 512, per-GPU batch: 16
- Noise schedule optimization: grid search over beta and omega at regular intervals
- clip_search_widths: [0.5, 0.6, 0.7, 0.8, 0.9]

---

## 3. LLaDA (ML-GSAI/LLaDA) — main branch

### GUIDELINES.md — Training code (verbatim)

```python
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

# The data is an integer tensor of shape (b, 4096),
# where b represents the batch size and 4096 is the sequence length.
input_ids = batch["input_ids"]

# 1% of pre-training data uses random length in [1, 4096]
if torch.rand(1) < 0.01:
    random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
    input_ids = input_ids[:, :random_length]

noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits

token_loss = F.cross_entropy(
    logits[masked_indices], input_ids[masked_indices], reduction='none'
) / p_mask[masked_indices]
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
```

### SFT code (verbatim from GUIDELINES.md)

```python
input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]
noisy_batch, _, p_mask = forward_process(input_ids)

# Do not add noise to the prompt
token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(
    noisy_batch.size(0), noisy_batch.size(1)
)
prompt_mask = (token_positions < prompt_length.unsqueeze(1))
noisy_batch[prompt_mask] = input_ids[prompt_mask]

answer_lengths = torch.sum((1 - prompt_mask.to(torch.int64)), dim=-1, keepdim=True)
answer_lengths = answer_length.repeat(1, noisy_batch.shape[1])

masked_indices = (noisy_batch == 126336)
logits = model(input_ids=noisy_batch).logits

token_loss = F.cross_entropy(
    logits[masked_indices], input_ids[masked_indices], reduction='none'
) / p_mask[masked_indices]
ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
```

### Hyperparameters from paper (arXiv 2502.09992)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight decay | 0.1 |
| Global batch size | 1280 |
| Per-GPU batch size | 4 |
| Sequence length | 4096 |
| Tokens per batch | 5.24M |
| Total training tokens | 2.3T |
| Compute | 0.13M H800 GPU hours |
| Mask token ID | 126336 |
| Random length sampling | 1% of data, uniform [1, 4096] |
| Sampling eps | 1e-3 |

**LR Schedule (Warmup-Stable-Decay):**

| Phase | LR | Token range |
|-------|----|-------------|
| Warmup | 0 -> 4e-4 | 0 - ~10B (2000 iters) |
| Stable | 4e-4 | ~10B - 1.2T |
| Crash recovery | 1e-4 (constant) | 1.2T - 2.0T |
| Final decay | 1e-4 -> 1e-5 (linear) | 2.0T - 2.3T |

**NOTE:** The "crash recovery" phase was unplanned — LLaDA hit NaN at 1.2T tokens. The
intended schedule was just warmup -> stable -> decay.

**SFT hyperparameters (from paper):**
- LR: 2.5e-5, WSD schedule, 50 steps warmup
- 3 epochs on 4.5M instruction pairs
- Mask token same as pretraining (126336)

---

## 4. Dream 7B (HKUNLP/Dream, also DreamLM/Dream) — main branch

### examples/run_sft_tulu3.sh (verbatim)

```bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: examples/run_sft_tulu3.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m src.trainer.fsdp_sft_trainer \
    diffusion.time_reweighting=cart \
    data.train_files=$HOME/data/tulu3/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.max_length=2048 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=2e-6 \
    data.micro_batch_size_per_gpu=8 \
    data.enable_perbatch_cutoff=True \
    data.perbatch_cutoff_type=random_with_input_pad \
    +data.perbatch_cutoff=True \
    model.partial_pretrain=Dream-org/Dream-v0-Base-7B \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=test_exp \
    trainer.project_name=diff-verl \
    trainer.experiment_name=test_exp \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 &
```

### src/trainer/fsdp_sft_trainer.py — optimizer setup (verbatim)

```python
self.optimizer = optim.AdamW(
    self.fsdp_model.parameters(),
    lr=self.config.optim.lr,
    betas=self.config.optim.betas,
    weight_decay=self.config.optim.weight_decay,
)

num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

self.lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=self.optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=self.total_steps,
)

# Gradient clipping
grad_norm = self.fsdp_model.clip_grad_norm_(
    max_norm=self.config.optim.clip_grad
)
```

### Extracted SFT config values

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 2e-6 |
| LR schedule | Cosine with warmup |
| Warmup | Ratio-based (exact ratio not in script) |
| Gradient clipping | Yes (value from config.optim.clip_grad, not in script) |
| Micro batch per GPU | 8 |
| Gradient checkpointing | True |
| Epochs | 3 |
| Seq length | 2048 |
| Loss weighting | CART (context-adaptive reweighted timestep) |
| Base model | Dream-org/Dream-v0-Base-7B (initialized from Qwen2.5-7B) |
| SFT data | Tulu 3 train split |
| Eval data | GSM8K test split |

### Pretraining details (from paper arXiv 2508.15487)

| Parameter | Value |
|-----------|-------|
| Base model init | Qwen2.5-7B weights |
| Total training tokens | 580B |
| Hardware | 96x H800 GPUs |
| Wall time | 256 hours |
| Data sources | Dolma v1.7, OpenCoder, DCLM-Baseline |
| Noise schedule | Linear alpha(t) = 1 - t |
| Special technique | CART (context-adaptive token-level noise rescheduling) |

**NOT DISCLOSED in paper:** optimizer betas, eps, weight decay, exact LR, warmup steps,
gradient clipping value, batch size for pretraining.

---

## 5. d1/diffu-GRPO (dllm-reasoning/d1) — main branch

### Hyperparameters from paper (arXiv 2504.12216)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| beta1 | 0.9 |
| beta2 | 0.99 |
| Weight decay | 0.1 |
| Learning rate | 3e-6 |
| Gradient clipping | 0.2 (max_grad_norm) |
| Per-GPU batch size | 6 |
| Gradient accumulation | 2 |
| Effective batch size | 96 (6 * 8 GPUs * 2 accum) |
| Hardware | 8x A100-80G |
| Flash Attention 2 | Yes |
| 4-bit quantization | Yes (QLoRA) |
| p_mask for log-prob | 0.15 |

### Training steps by task

| Task | Steps |
|------|-------|
| GSM8K | 7700 |
| MATH500 | 6600 |
| Countdown | 5000 |
| Sudoku | 3800 |

### SFT stage (from README)

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ddp_config.yaml \
    --main_process_port 29500 \
    --num_processes 2 \
    sft_train.py \
    --grad_accum_steps 4 \
    --batch_size 1 \
    --num_epochs 20
```

| Parameter | Value |
|-----------|-------|
| Per-GPU batch | 1 |
| GPUs | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Epochs | 20 |

### GRPO-specific parameters (from paper)

- K = 6 rollouts per prompt (some experiments use K=8)
- n_mc = 8 MC samples per rollout for log-probability estimation
- Total forward passes per step: K * n_mc = 48-64
- 12 policy gradient inner update iterations per batch
- LoRA rank: 128, alpha: 64, dropout: 0.05
- Remasking strategy: low_confidence
- Sequence length: 256 tokens (shorter than pretraining)

---

## 6. dllm unified library (ZHZisZZ/dllm) — main branch

### examples/llada/sft.py — training arguments (verbatim)

```python
@dataclass
class TrainingArguments:
    output_dir: str = ".models/LLaDA-8B-Base/tulu-3-sft-mixture[train:10000,test:1000]"
    group_by_length: bool = True
    num_train_epochs: int = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4

@dataclass
class ModelArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"

@dataclass
class DataArguments:
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = True
```

### Accelerate configs

- `scripts/accelerate_configs/zero2.yaml` — ZeRO-2 distributed training
- `scripts/accelerate_configs/fsdp.yaml` — FSDP config

### Launch command

```bash
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml examples/llada/sft.py
```

### Key values

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Per-GPU batch | 4 |
| Epochs | 5 |
| Base model | GSAI-ML/LLaDA-8B-Base |
| Mask prompt loss | True |
| Group by length | True |
| Framework | HF Trainer + Accelerate |

---

## 7. Fast-dLLM v2 (NVlabs/Fast-dLLM) — main branch

### Hyperparameters from paper (arXiv 2509.26328)

| Parameter | 1.5B model | 7B model |
|-----------|-----------|---------|
| Optimizer | AdamW | AdamW |
| Learning rate | 2e-5 | 1e-5 |
| Warmup | 500 steps (linear) | 500 steps (linear) |
| Total steps | 6000 | 2500 |
| Total tokens | ~3.15B | ~1.31B |
| Batch size (global) | 256 | 256 |
| Seq length | 2048 | 2048 |
| Tokens per step | 524K | 524K |
| Block size | 32 | 32 |
| Sub-block size | 8 | 8 |
| Hardware | 64x A100 | 64x A100 |
| Distributed strategy | DeepSpeed Zero-3 | DeepSpeed Zero-3 |
| Wall time | ~8h | ~12h |
| Base model | Qwen2.5-1.5B | Qwen2.5-7B |
| Training framework | LMFlow | LMFlow |

**NOT DISCLOSED:** weight decay, gradient clipping, beta1, beta2, eps.

---

## Cross-Repo Comparison Table

| | MDLM | BD3LM | LLaDA | Dream SFT | d1 GRPO | Fast-dLLM v2 | dllm SFT |
|---|---|---|---|---|---|---|---|
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | HF default |
| **LR** | 3e-4 | 3e-4 | 4e-4 | 2e-6 | 3e-6 | 1-2e-5 | 2e-5 |
| **beta1** | 0.9 | 0.9 | n/a | n/a | 0.9 | n/a | n/a |
| **beta2** | 0.999 | 0.999 | n/a | n/a | 0.99 | n/a | n/a |
| **eps** | 1e-8 | 1e-8 | n/a | n/a | n/a | n/a | n/a |
| **Weight decay** | 0 | 0 | 0.1 | n/a | 0.1 | n/a | n/a |
| **EMA** | 0.9999 | 0.9999 | n/a | n/a | n/a | n/a | n/a |
| **Grad clip** | 1.0 | 1.0 | n/a | 0.2 | n/a | n/a | n/a |
| **Batch (global)** | 512 | 512 | 1280 | 64 | 96 | 256 | n/a |
| **Seq len** | 1024 | 1024 | 4096 | 2048 | 256 | 2048 | n/a |
| **Max steps** | 1M | 1M | n/a | 3 epochs | 3.8-7.7K | 2.5-6K | 5 epochs |
| **Precision** | bf16 | bf16 | n/a | n/a | n/a | n/a | n/a |
| **Warmup** | 2500 steps | 2500 steps | 2000 steps | ratio-based | n/a | 500 steps | n/a |
| **LR sched** | constant | constant | WSD | cosine | constant | n/a | n/a |
| **Antithetic** | Yes | Yes | Yes | n/a | Yes | n/a | n/a |
| **Framework** | Lightning | Lightning | custom | FSDP/verl | Accelerate | LMFlow/DS | HF Trainer |

---

## Notable Training Tricks (verbatim from sources)

### Random length sampling (LLaDA)
1% of pretraining data uses random sequence length uniformly sampled from [1, 4096].
Prevents the model from only learning to denoise full-length sequences.

### CART loss weighting (Dream)
`diffusion.time_reweighting=cart` — Context-Adaptive Reweighted Timestep. Replaces
standard ELBO weighting with a learned, context-dependent weighting that reduces
variance. Appears to allow 10x lower LR (2e-6 vs 2.5e-5 for standard ELBO SFT).

### Antithetic timestep sampling (MDLM, BD3LM, LLaDA, d1)
`t_i = (u + i/B) mod 1` where u ~ Uniform(0,1). Reduces variance of ELBO gradient
estimator. Universal across all implementations that disclose this detail.

### EMA (MDLM, BD3LM)
Decay rate 0.9999. Applied to model weights. Used for evaluation and generation.
LLaDA and Dream do not report using EMA.

### 4-bit QLoRA (d1/diffu-GRPO)
Base model quantized to 4-bit, LoRA adapters trained in bf16/fp16.
LoRA config: rank=128, alpha=64, dropout=0.05. High rank needed for reasoning.

### Zero weight decay (MDLM/BD3LM)
MDLM and BD3LM use weight_decay=0, which is unusual. This may work because they
use a DiT backbone rather than a standard Transformer decoder. LLaDA and d1 use
the standard 0.1 weight decay with decoder-based architectures.

---

## Sources

- [MDLM config.yaml (GitHub raw)](https://raw.githubusercontent.com/kuleshov-group/mdlm/master/configs/config.yaml)
- [LLaDA GUIDELINES.md (GitHub raw)](https://raw.githubusercontent.com/ML-GSAI/LLaDA/main/GUIDELINES.md)
- [Dream run_sft_tulu3.sh (GitHub raw)](https://raw.githubusercontent.com/HKUNLP/Dream/main/examples/run_sft_tulu3.sh)
- [Dream fsdp_sft_trainer.py (GitHub raw)](https://raw.githubusercontent.com/HKUNLP/Dream/main/src/trainer/fsdp_sft_trainer.py)
- [dllm sft.py (GitHub raw)](https://raw.githubusercontent.com/ZHZisZZ/dllm/main/examples/llada/sft.py)
- [LLaDA paper (arXiv 2502.09992)](https://arxiv.org/abs/2502.09992)
- [MDLM paper (arXiv 2406.07524)](https://arxiv.org/abs/2406.07524)
- [BD3LM paper (arXiv 2503.09573)](https://arxiv.org/abs/2503.09573)
- [Dream 7B paper (arXiv 2508.15487)](https://arxiv.org/abs/2508.15487)
- [d1/diffu-GRPO paper (arXiv 2504.12216)](https://arxiv.org/abs/2504.12216)
- [Fast-dLLM v2 paper (arXiv 2509.26328)](https://arxiv.org/abs/2509.26328)
