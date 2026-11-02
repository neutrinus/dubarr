# Stage 1: Builder
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 as builder

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_HTTP_TIMEOUT=600

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake python3-pip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv-python

WORKDIR /build

# One Python 3.12 venv for everything: API, Whisper, pyannote, Demucs, llama-cpp and the OmniVoice TTS server
# (the TTS server still runs as its own process, but on this same interpreter).
RUN uv python install 3.12 && \
    uv venv /app/.venv --python 3.12 && \
    uv pip install --no-cache-dir --python /app/.venv/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy==2.2.2" "torch==2.8.0" "torchvision" "torchaudio==2.8.0" \
    "pyannote.audio==4.0.7" "faster-whisper" "av<19" "demucs>=4.1.0" "diffq" \
    "omnivoice==0.2.1" "flask" \
    "huggingface_hub" "pydub" "soundfile" "humanfriendly" "psutil" "scipy" "requests" "syllables" \
    "fastapi" "uvicorn[standard]" "websockets" "jinja2" "python-multipart" && \
    uv pip install --no-cache-dir --python /app/.venv/bin/python3 \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 && \
    uv cache clean

# Optional legacy XTTS v2 backend (TTS_BACKEND=xtts): `docker build --build-arg WITH_XTTS=1 .`
# Coqui needs transformers<5 while OmniVoice needs >=5.3, so XTTS lives in its own venv when it is wanted at all.
ARG WITH_XTTS=0
RUN mkdir -p /app/.venv_xtts && if [ "$WITH_XTTS" = "1" ]; then \
    uv python install 3.10 && \
    uv venv --allow-existing /app/.venv_xtts --python 3.10 && \
    uv pip install --no-cache-dir --python /app/.venv_xtts/bin/python3 \
    --index-strategy unsafe-best-match \
    "numpy>=1.26.0" "torch==2.8.0" "torchaudio==2.8.0" "transformers>=4.57,<5" \
    "flask" "coqui-tts==0.28.0" && \
    uv cache clean && \
    f=/app/.venv_xtts/lib/python3.10/site-packages/TTS/tts/layers/tortoise/autoregressive.py && \
    sed -i 's/from transformers.pytorch_utils import isin_mps_friendly as isin/import torch\n\ndef isin(elements, test_elements, *args, **kwargs):\n    return torch.isin(elements, test_elements)/' "$f"; \
    fi

# Stage 2: Final
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget aria2 libsndfile1 gosu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python installations and built venvs from builder
COPY --from=builder /usr/local/uv-python /usr/local/uv-python
COPY --from=builder /app/.venv /app/.venv
# Empty unless built with WITH_XTTS=1
COPY --from=builder /app/.venv_xtts /app/.venv_xtts

# Copy the project files
COPY . .

# Environment setup
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:/app/src"

WORKDIR /app/src

# Use entrypoint script to handle PUID/PGID
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python3", "server.py"]
