# Base image with CUDA 13.1.1 support, based on Ubuntu 24.04
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential system dependencies including Python 3.12 (default in 24.04), ffmpeg, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    aria2 \
    libsndfile1 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create virtual environment and set it as default
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Ensure nvcc and CUDA libs are in path for compilation
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"

# Clone F5-TTS repo with all tags and checkout latest stable version
RUN git clone --no-single-branch https://github.com/SWivid/F5-TTS.git /app/F5-TTS && \
    cd /app/F5-TTS && \
    git checkout 1.1.15

# Install EVERYTHING in one go to ensure consistent dependency resolution.
# We pin numpy to 2.2.2 to satisfy numba and pyannote.
# We pin pyannote.audio to 4.0.4.
# CMAKE_ARGS ensures llama-cpp-python has CUDA support.
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN uv pip install --no-cache-dir \
    "numpy==2.2.2" \
    "torch>=2.8.0" \
    "torchvision" \
    "torchaudio>=2.8.0" \
    "pyannote.audio==4.0.4" \
    "demucs" \
    "diffq" \
    "faster-whisper" \
    "hf_transfer" \
    "syllables" \
    "humanfriendly" \
    "psutil" \
    "safetensors" \
    --no-binary llama-cpp-python "llama-cpp-python" \
    "/app/F5-TTS" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match

# Create and set the working directory inside the container
WORKDIR /app

# Ensure F5-TTS src is in PYTHONPATH
ENV PYTHONPATH="/app:/app/F5-TTS/src"

# Copy the application files into the container
COPY . .

# Verification step: ensure core libraries are importable
RUN python -c "import torch; import faster_whisper; import TTS; print('Environment verification successful')"

# Use entrypoint script to handle PUID/PGID
ENTRYPOINT ["/app/entrypoint.sh"]

# Define the default command to be executed when the container starts
CMD ["python3", "main.py"]