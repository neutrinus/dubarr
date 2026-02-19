# Stage 1: Builder
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 as builder

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_HTTP_TIMEOUT=600

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake python3-pip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv-python

WORKDIR /build

# 1. Install Python versions and setup Venvs
RUN uv python install 3.12 3.10 && \
    uv venv /app/.venv_app --python 3.12 && \
    uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy==2.2.2" "torch>=2.5.0" "torchvision" "torchaudio" \
    "pyannote.audio==4.0.4" "faster-whisper" "demucs" "diffq" \
    "huggingface_hub[hf_transfer]" "pydub" "soundfile" "humanfriendly" "psutil" "scipy" "requests" "syllables" \
    "fastapi" "uvicorn[standard]" "websockets" "jinja2" "python-multipart" && \
    uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

RUN uv venv /app/.venv_tts --python 3.10 && \
    uv pip install --no-cache-dir --python /app/.venv_tts/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy<2.0" "torch==2.4.0" "torchaudio==2.4.0" \
    "flask" "git+https://github.com/idiap/coqui-ai-TTS.git"

# Fix the transformers breaking change in XTTS
RUN if [ -f /app/.venv_tts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py ]; then \
    sed -i 's/from transformers.pytorch_utils import isin_mps_friendly as isin/import torch\n\ndef isin(elements, test_elements, *args, **kwargs):\n    return torch.isin(elements, test_elements)/' \
    /app/.venv_tts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py; \
    fi

# Stage 2: Final
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/hf_cache

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget aria2 libsndfile1 gosu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built venvs from builder
COPY --from=builder /app/.venv_app /app/.venv_app
COPY --from=builder /app/.venv_tts /app/.venv_tts

# Copy the project files
COPY . .

# Environment setup
ENV PATH="/app/.venv_app/bin:$PATH"
ENV PYTHONPATH="/app:/app/src"

WORKDIR /app/src

# Use entrypoint script to handle PUID/PGID
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python3", "server.py"]
