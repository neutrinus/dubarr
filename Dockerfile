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

# Stage 2: Final
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Don't enable hf_transfer here: OmniVoice's venv doesn't ship it, and huggingface_hub 2.x uses hf_xet anyway.
ENV HF_HOME=/app/hf_cache
# CTranslate2/faster-whisper loads CUDA 12 cuBLAS; the pinned OmniVoice torch ships it.
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget aria2 libsndfile1 gosu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python installations and built venvs from builder
COPY --from=builder /usr/local/uv-python /usr/local/uv-python
COPY --from=builder /app/.venv /app/.venv

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
