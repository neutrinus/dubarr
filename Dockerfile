# Base image with CUDA 12.1 support, based on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential system dependencies including Python, pip, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    aria2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Install hf_transfer for high-speed Hugging Face downloads
RUN pip install --no-cache-dir hf_transfer syllables humanfriendly psutil

# Install PyTorch compatible with CUDA 12.1. This is a crucial step for most AI libraries.
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install project-specific Python libraries

# 1. Demucs for audio source separation
RUN pip install --no-cache-dir -U demucs

# 2. Pyannote.audio for speaker diarization.
#    Note: pyannote.audio 3.1 requires accepting user agreements on Hugging Face.
#    A User Access Token with read permissions is required during execution.
RUN pip install --no-cache-dir "huggingface-hub<0.24" pyannote.audio==3.1.1
# Install diffq for mdx_extra_q demucs model
RUN pip install --no-cache-dir diffq

# 3. Faster-Whisper for Automatic Speech Recognition (ASR)
RUN pip install --no-cache-dir faster-whisper

# 4. Llama-cpp-python for running the GGUF translation model.
#    The CMAKE_ARGS environment variable is set to build the library with CUDA support (cuBLAS).
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir llama-cpp-python --force-reinstall --upgrade --verbose

# 5. Coqui TTS for Text-to-Speech synthesis (XTTS v2 model)
RUN pip install --no-cache-dir transformers==4.40.0
RUN pip install --no-cache-dir TTS

# Create and set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . .

# Verification step: ensure core libraries are importable
RUN python -c "import torch; import faster_whisper; import TTS; print('Environment verification successful')"

# Define the default command to be executed when the container starts
CMD ["python", "main.py"]
