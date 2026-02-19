FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/hf_cache
ENV UV_HTTP_TIMEOUT=600

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    ffmpeg \
    git \
    wget \
    aria2 \
    libsndfile1 \
    cmake \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv-python

WORKDIR /app

# 1. Install Python versions
RUN uv python install 3.12 3.10

# 2. Setup App Venv (Modern Stack)
RUN uv venv /app/.venv_app --python 3.12 && \
    uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy==2.2.2" "torch>=2.5.0" "torchvision" "torchaudio" \
    "pyannote.audio==4.0.4" "faster-whisper" "demucs" "diffq" \
    "huggingface_hub[hf_transfer]" "pydub" "soundfile" "humanfriendly" "psutil" "scipy" "requests" "syllables" \
    "fastapi" "uvicorn" "jinja2" "python-multipart" && \
    uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 && \
    uv cache clean

# 3. Setup TTS Venv (Legacy Stack for XTTS v2)
RUN uv venv /app/.venv_tts --python 3.10 && \
    uv pip install --no-cache-dir --python /app/.venv_tts/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy<2.0" "torch==2.4.0" "torchaudio==2.4.0" \
    "flask" "git+https://github.com/idiap/coqui-ai-TTS.git" && \
    uv cache clean

# Fix the transformers breaking change in XTTS
RUN if [ -f /app/.venv_tts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py ]; then \
    sed -i 's/from transformers.pytorch_utils import isin_mps_friendly as isin/import torch\n\ndef isin(elements, test_elements, *args, **kwargs):\n    return torch.isin(elements, test_elements)/' \
    /app/.venv_tts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py; \
    fi

# 4. Pre-download models (Basic setup only)
# XTTS v2 (Coqui) - We'll trigger the download during build to bake it in
RUN mkdir -p /app/tts_cache && \
    export COQUI_TOS_AGREED=1 && \
    export TTS_HOME=/app/tts_cache && \
    /app/.venv_tts/bin/python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

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
