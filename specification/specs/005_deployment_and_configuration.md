# Spec 005: Deployment & Configuration

## 1. Deployment Model
Dubarr is designed to run as a containerized application, primarily orchestrated via **Docker Compose**. This aligns with the "Arr" stack philosophy (Radarr, Sonarr, etc.) for easy deployment in home server environments.

### 1.1 Containerization
-   **Base Image:** Uses a slim Ubuntu-based NVIDIA CUDA image (`nvidia/cuda:13.1.1-devel-ubuntu24.04`) for hardware acceleration.
-   **Dual-Environment Strategy:** The container hosts two distinct Python virtual environments to resolve dependency conflicts:
    1.  `/app/.venv_app` (Python 3.12): Runs the main API, Whisper, Pyannote, and FFmpeg logic.
    2.  `/app/.venv_tts` (Python 3.10): Runs the Coqui-TTS engine (XTTS v2) as an internal microservice on port 5050.
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
      # Optional: Set a specific GPU for TTS if needed
      - GPU_TTS=0
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
| `MOCK_MODE` | If `true`, bypasses ML models and returns dummy data. For testing/dev. | `false` |
| `GPU_TTS` | Specific GPU ID (integer) for the TTS microservice. | `0` |

### 2.3 GPU Resource Management
The system includes an **Intelligent GPU Manager** that orchestrates VRAM usage using a **Lazy Loading** and **Memory Rotation** strategy.

-   **Device Mapping:** The system uses hardware signatures (Name + VRAM size) to correctly map `nvidia-smi` system indices to PyTorch `cuda:X` indices, preventing misallocation on multi-GPU systems.
-   **Behavior:**
    -   Models (LLM, TTS, Whisper, Diarization) are loaded into memory only when their specific stage is active.
    -   Whisper and Diarization models are explicitly unloaded immediately after use to free space for synthesis.
    -   **Parallel Slots:** LLM (Gemma 3 12B) is configured with multiple parallel inference slots (e.g., 4) when sufficient VRAM is available.
    -   **RPC Serialization:** Access to GPU resources is serialized via an Internal RPC layer with priority-based queuing.

## 3. Hardware Requirements
-   **GPU:** Strongly recommended (NVIDIA CUDA). 
-   **Multi-GPU Recommendation:**
    -   **Primary (12GB+):** LLM (Gemma 12B) requires ~10GB for 8k context and parallel slots.
    -   **Secondary (8GB+):** TTS (XTTS) and Analysis (Whisper/Diarization) work optimally here.
-   **RAM:** Minimum 16GB.

## 4. Local Development Notes
Running Dubarr locally (outside Docker) requires creating two separate virtual environments:
1.  **Main Env (Py3.12):** `uv sync` from `pyproject.toml`.
2.  **TTS Env (Py3.10):** Must be created manually to install `coqui-tts`. Run `src/tts_server.py` inside this environment on port 5050 before starting the main app.
