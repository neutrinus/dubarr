# Spec 005: Deployment & Configuration

## 1. Deployment Model
Dubarr is designed to run as a containerized application, primarily orchestrated via **Docker Compose**. This aligns with the "Arr" stack philosophy (Radarr, Sonarr, etc.) for easy deployment in home server environments.

### 1.1 Containerization
-   **Image:** The application is packaged as a Docker image containing all dependencies (Python, FFmpeg, system libraries).
-   **Base Image:** Uses a slim Python base image (e.g., `python:3.10-slim`) optimized for size and security.
-   **Volumes:**
    -   `/app/videos`: Mapped to the host's video storage (Input/Output).
    -   `/app/models`: Mapped to persist large ML models (Whisper, TTS) across restarts.
    -   `/app/config`: Mapped for configuration files (optional) and database (`dubarr.db`).

### 1.2 Docker Compose
The recommended deployment method is via `docker-compose.yml`.

```yaml
services:
  dubarr:
    image: ghcr.io/neutrinus/dubarr:latest
    container_name: dubarr
    environment:
      - OPENAI_API_KEY=sk-...
      - HF_TOKEN=hf_...
    volumes:
      - /path/to/media:/app/videos
      - /path/to/models:/app/models
      - /path/to/config:/app/config
    ports:
      - "8000:8000"
    restart: unless-stopped
```

## 2. Configuration
Configuration is managed strictly through **Environment Variables**. This follows the 12-Factor App methodology.

### 2.1 Core Settings
| Variable | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | **Required.** Key for OpenAI API (LLM). | - |
| `HF_TOKEN` | **Required.** Hugging Face token for downloading Pyannote/Whisper models. | - |
| `API_USER` | Username for Basic Auth. | `admin` |
| `API_PASS` | Password for Basic Auth. | `admin` |
| `TARGET_LANGS` | Comma-separated list of default target languages. | `pl` |

### 2.2 Advanced Settings
| Variable | Description | Default |
| :--- | :--- | :--- |
| `DEBUG_MODE` | If `true`, keeps intermediate files (wav segments, json) for debugging. | `false` |
| `TTS_MODEL` | Selects the TTS engine (e.g., `xtts`, `openai`, `elevenlabs`). | `xtts` |
| `WHISPER_MODEL` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). | `medium` |
| `DEVICE` | Computation device (`cuda`, `cpu`, `mps`). Auto-detected if unset. | `cuda` |

## 3. Hardware Requirements
-   **GPU:** Strongly recommended (NVIDIA CUDA) for reasonable processing times (Whisper + TTS).
-   **CPU-only:** Supported but significantly slower (not recommended for production).
-   **RAM:** Minimum 8GB (16GB recommended for larger Whisper models).
