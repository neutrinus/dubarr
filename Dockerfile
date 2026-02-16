FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/hf_cache
ENV TTS_HOME=/app/tts_cache

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

WORKDIR /app

# 1. Install Python versions
RUN uv python install 3.12
RUN uv python install 3.10

# 2. Setup App Venv (Modern Stack)
RUN uv venv /app/.venv_app --python 3.12

RUN uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy==2.2.2" "torch>=2.5.0" "torchvision" "torchaudio"

RUN uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    "pyannote.audio==4.0.4" "faster-whisper" "demucs" "diffq"

RUN uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    "huggingface_hub[hf_transfer]" "pydub" "soundfile" "humanfriendly" "psutil" "scipy" "requests" "syllables"

RUN uv pip install --no-cache-dir --python /app/.venv_app/bin/python3 \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# 3. Setup TTS Venv (Legacy Stack for XTTS v2)
RUN uv venv /app/.venv_tts --python 3.10

# Layer 3a: Base ML for TTS
RUN uv pip install --no-cache-dir --python /app/.venv_tts/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy<2.0" "torch==2.4.0" "torchaudio==2.4.0"

# Layer 3b: Common libs
RUN uv pip install --no-cache-dir --python /app/.venv_tts/bin/python3 \
    "transformers<=4.43.3" "pydantic<2.0" "flask"

# Layer 3c: XTTS fork
RUN uv pip install --no-cache-dir --python /app/.venv_tts/bin/python3 \
    "git+https://github.com/idiap/coqui-ai-TTS.git"

# Fix the transformers breaking change in XTTS
RUN if [ -f /app/.venv_tts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py ]; then \
    sed -i 's/from transformers.pytorch_utils import isin_mps_friendly as isin/import torch\n\ndef isin(a, b, *args, **kwargs):\n    return torch.isin(a, b)/' \
    /app/.venv_tts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py; \
    fi

# 4. Pre-download models
# Gemma 3 12B
RUN mkdir -p /app/models && /app/.venv_app/bin/python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='bartowski/google_gemma-3-12b-it-GGUF', filename='google_gemma-3-12b-it-Q4_K_M.gguf', local_dir='/app/models')"

# XTTS v2 (Coqui) - We'll trigger the download during build to bake it in
RUN mkdir -p /app/tts_cache && \
    export COQUI_TOS_AGREED=1 && \
    export TTS_HOME=/app/tts_cache && \
    /app/.venv_tts/bin/python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Copy the project files
COPY . .

# Environment setup
ENV PATH="/app/.venv_app/bin:$PATH"
ENV PYTHONPATH="/app"

# Use entrypoint script to handle PUID/PGID
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python3", "server.py"]
