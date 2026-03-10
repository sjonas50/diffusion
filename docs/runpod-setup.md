# RunPod GPU Training Setup

## Overview

This document covers deploying and training diffusion LLMs on RunPod GPU pods. The setup has been
validated on A100-SXM4-80GB with Qwen3-0.6B masked diffusion training.

---

## Quick Start

### 1. Create Pod

Use the RunPod MCP tools or web console:

- **GPU**: A100-SXM4-80GB (recommended), A100-PCIe-80GB, or H100
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container disk**: 20 GB
- **Volume**: 50 GB at `/workspace`
- **Ports**: `22/tcp` (SSH), `8888/http` (Jupyter, optional)
- **Environment**: Set `PUBLIC_KEY` to your SSH public key content

### 2. SSH In

```bash
ssh -p <PORT> root@<HOST> -i ~/.ssh/id_ed25519
```

The port and host are shown in the RunPod pod details under "SSH over exposed TCP".

### 3. Install Dependencies

```bash
# Clone repo
cd /workspace
git clone https://github.com/sjonas50/diffusion.git
cd diffusion

# Install with system pip — DO NOT use uv venv (see Gotchas below)
pip install -e ".[dev]"

# Install Flash Attention 2 — use regular pip for prebuilt wheels
pip install flash-attn
```

### 4. Run Training

```bash
mkdir -p /workspace/logs /workspace/checkpoints

nohup python3 scripts/pretrain.py \
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
  --dataloader_num_workers 4 \
  > /workspace/logs/training.log 2>&1 &
```

### 5. Monitor Training

```bash
# Live progress bar
tail -f /workspace/logs/training.log

# Check GPU usage
nvidia-smi

# Check loss values (after first checkpoint)
python3 -c "
import json
state = json.load(open('/workspace/checkpoints/finance-qwen3-a100/checkpoint-500/trainer_state.json'))
for e in state['log_history']:
    if 'loss' in e:
        print(f\"step {e['step']:5d}  loss={e['loss']:.4f}  grad_norm={e.get('grad_norm', 'N/A')}  lr={e.get('learning_rate', 'N/A')}\")
"
```

---

## Gotchas and Lessons Learned

### Docker Image Tags

**Use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.**

RunPod Docker image tags do NOT follow a simple `<pytorch_version>-<python_version>-<cuda_version>`
pattern. Many tags that look valid don't exist. For example,
`runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04` returns a `manifest not found`
error. Always verify the tag exists before creating a pod.

### DO NOT Use `uv venv` on RunPod

RunPod images ship with PyTorch pre-installed at the system level, compiled against the specific
CUDA version on the machine. Creating a `uv venv` and running `uv pip install -e ".[dev]"` will:

1. Install PyTorch from PyPI (e.g., `2.10.0+cu128`) instead of using the system's pre-installed
   version (e.g., `2.4.1+cu124`)
2. The PyPI PyTorch may target a different CUDA version than the machine's driver
3. The venv also won't include `pip`, making it hard to install flash-attn

**Correct approach:** Install directly with system pip:
```bash
pip install -e ".[dev]"
```

### Flash Attention Installation

**Use `pip install flash-attn` (regular pip), NOT `uv pip install flash-attn --no-build-isolation`.**

- Regular `pip` finds prebuilt wheels matching the system's PyTorch+CUDA versions and installs in seconds
- `uv pip install` with `--no-build-isolation` triggers a source build that takes 20-40 minutes
- `uv pip install` without `--no-build-isolation` may fail to find compatible builds

### HF Trainer Loss Logging

HF Trainer (transformers 5.3.0) logs loss values via tqdm progress bar updates. When output is
redirected to a file with `nohup ... > log.log 2>&1`, the carriage return (`\r`) overwrites
previous output in the log file, making loss values invisible.

**Workarounds:**
1. Check `trainer_state.json` inside checkpoint directories — it contains the full loss history
2. Add `--report_to wandb` and use W&B dashboard for live monitoring
3. Use `tr '\r' '\n'` when processing the log: `cat training.log | tr '\r' '\n' | grep loss`
4. Consider adding a custom logging callback that writes to a separate file

### Double Backbone Initialization

`MaskedDiffusionLM.from_configs()` loads the backbone twice — once in `BidirectionalTransformer.__init__`
and once implicitly. Not a correctness issue but doubles initialization time and peak memory during
startup. For large models (7B+), this can take significant time and may OOM on smaller GPUs.

### SSH Key Setup

Set the SSH public key via the `PUBLIC_KEY` environment variable when creating the pod. The pod may
take 1-2 minutes after creation before SSH is available. If you get "Connection refused", wait and
retry.

---

## Validated Training Configurations

### Qwen3-0.6B on A100-SXM4-80GB (March 2026)

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-0.6B (595M params) |
| Process | Masked diffusion, linear schedule |
| Attention | flash_attention_2 (28 modules patched is_causal=False) |
| Dataset | ashraq/financial-news-articles (306k articles, 375k blocks of 512 tokens) |
| Batch size | 8 per device, 4 grad accum steps (effective 32) |
| LR | 3e-5, cosine decay, 200 warmup steps |
| Precision | bf16 |
| VRAM | ~13.9 GB / 80 GB |
| GPU util | 93-100% |
| Speed | ~1.06 s/iter |
| Cost | ~$1.49/hr ($0.50 total for 2000 steps) |

### Hyperparameter Rationale

- **LR 3e-5**: From `docs/training-recipes.md` — recommended for 600M AR adaptation (20-40x lower than from-scratch)
- **block_size 2048**: Matches Qwen3 pretrained context length. Use 512 for quick smoke tests.
- **batch_size 16 x grad_accum 8**: Effective batch of 128 sequences = 262K tokens/step.
- **warmup 500**: 10% of total steps (standard for adaptation)
- **weight_decay 0.1**: Standard across all dLLM papers (LLaDA, d1)
- **max_grad_norm 1.0**: Standard. Prevents NaN crashes from ELBO-weighted loss

### VRAM Limits (Qwen3-0.6B, 151K vocab, A100-80GB)

The CE loss `logits.reshape(B*L, V)` dominates VRAM due to Qwen3's large vocab (151,670):

| batch | block_size | CE tensor | Total VRAM | Status |
|-------|-----------|-----------|------------|--------|
| 8 | 512 | 2.4 GB | ~14 GB | OK |
| 16 | 2048 | 18.8 GB | ~40 GB | OK |
| 32 | 2048 | 37.6 GB | ~80 GB | **OOM** |

**Use batch_size=16 with block_size=2048** on A100-80GB. Do NOT use batch_size=32.

### Previous Training Run Issues (MPS, Pre-A100)

An earlier training run on MPS (finance-qwen3, 1500 steps) showed:
- Good convergence for first 500 steps: loss 25.35 -> 9.30
- Catastrophic instability after step 500 due to `--ignore_optimizer_on_resume` (optimizer state reset)
- Loss oscillated between 10-51, grad norms spiked to 6449
- **Lesson**: Never reset optimizer state mid-training unless switching optimizers entirely

---

## Cost Optimization

- **Spot instances**: Use RunPod community cloud for ~50% savings. Set `--save_steps` aggressively (every 500 steps) to preserve progress if preempted.
- **Stop when idle**: Use `runpod.stop_pod()` or the MCP tool between training runs
- **Right-size GPU**: Qwen3-0.6B only uses ~14GB VRAM. An A40 (48GB) or even RTX 4090 (24GB) would suffice and cost less. A100 is overkill for this model size.
- **Volume persistence**: Use a network volume to persist checkpoints across pod restarts

---

## Retrieving Checkpoints

```bash
# From local machine — download checkpoint
scp -P <PORT> -i ~/.ssh/id_ed25519 -r \
  root@<HOST>:/workspace/checkpoints/finance-qwen3-a100/checkpoint-2000 \
  ./checkpoints/finance-qwen3-a100/

# Or push to HuggingFace Hub from the pod
pip install huggingface_hub
huggingface-cli login
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/workspace/checkpoints/finance-qwen3-a100/checkpoint-2000',
    repo_id='your-org/qwen3-0.6b-diffusion-finance',
    repo_type='model',
)
"
```
