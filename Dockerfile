# Multi-stage Dockerfile for diffusion-lm training framework
# Uses PyTorch 2.4+ with Python 3.11 and CUDA 12.1

# ---- Build stage: install flash-attn (needs CUDA headers) ----
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install project and dependencies (base extras only — flash-attn added below)
RUN uv pip install --system -e "." --no-cache

# Install flash-attn (requires CUDA build environment)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn installation failed; falling back to eager attention"


# ---- Final stage ----
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS final

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for final stage package management
RUN pip install --no-cache-dir uv

# Copy source and pyproject for editable install in final stage
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package
RUN uv pip install --system -e "." --no-cache

# Copy scripts and configs
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create directories for checkpoints and data
RUN mkdir -p /app/checkpoints /app/data

# Default environment
ENV PYTHONPATH=/app/src
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/datasets
ENV TOKENIZERS_PARALLELISM=false

# Verify installation
RUN python -c "import diffusion_lm; print('diffusion_lm installed OK')"

CMD ["python", "scripts/pretrain.py", "--help"]
