# Spec 005: Deployment & Configuration

## 1. Deployment Model
Dubarr is designed to run as a containerized application, primarily orchestrated via **Docker Compose**. This aligns with the "Arr" stack philosophy (Radarr, Sonarr, etc.) for easy deployment in home server environments.

### 1.1 Containerization
-   **Base Image:** Uses a slim Ubuntu-based NVIDIA CUDA image (`nvidia/cuda:13.1.1-devel-ubuntu24.04`) for hardware acceleration.
-   **Single Environment:** The container has one Python 3.12 virtual environment, `/app/.venv`, with the API, Whisper,
    Pyannote, Demucs, llama-cpp and the OmniVoice TTS engine. The TTS engine still runs as an internal microservice
    (a separate process on port 5051, started by the app on the same interpreter), so a TTS crash or restart does not
    take the worker down.
-   **Optional legacy XTTS:** Coqui-TTS (XTTS v2) needs `transformers<5`, which conflicts with OmniVoice. It is only
    installed when the image is built with `--build-arg WITH_XTTS=1`, into a separate Python 3.10 venv
    (`/app/.venv_xtts`, port 5050), and selected with `TTS_BACKEND=xtts`.
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
| `LLM_API_URL` | Base URL of an OpenAI-compatible server (e.g. llama.cpp `llama-server`, `http://host:8080/v1`). When set, no LLM is loaded in-process and its VRAM stays free. | - |
| `LLM_API_MODEL` / `LLM_API_KEY` | Model name and key sent to that server (llama-server ignores both). | - |
| `LLM_TIMEOUT` | Seconds per request to the LLM server. | `600` |
| `METADATA_LOOKUP` | Show/movie context for the LLM: `on` (`.nfo` files next to the video, then TVmaze for episodes and TMDB for movies), `local` (`.nfo` only), `off`. | `on` |
| `TMDB_API_KEY` | Optional TMDB API key (v3 key or v4 read token) for movie metadata. Episodes use TVmaze, which needs no account. | - |
| `METADATA_TIMEOUT` | Seconds per metadata request. | `10` |
| `LLM_N_CTX` | Context size of the in-process model. | `8192` |

### 2.3 GPU Resource Management
The system includes an **Intelligent GPU Manager** that orchestrates VRAM usage using a **Lazy Loading** and **Memory Rotation** strategy.

-   **Device Mapping:** The system uses hardware signatures (Name + VRAM size) to correctly map `nvidia-smi` system indices to PyTorch `cuda:X` indices, preventing misallocation on multi-GPU systems.
-   **Behavior:**
    -   Models (LLM, TTS, Whisper, Diarization) are loaded into memory only when their specific stage is active.
    -   Whisper and Diarization models are explicitly unloaded immediately after use to free space for synthesis.
    -   **LLM:** All LLM calls are chat completions with JSON output, so the model's own chat template is used. With `LLM_API_URL` the LLM runs in an external server (e.g. an existing llama.cpp container) and dubarr loads no LLM itself.
    -   **Parallel Slots:** LLM (Gemma 3 12B) utilizes parallel inference slots when sufficient VRAM is available. However, due to stability requirements with `llama-cpp-python` CUDA bindings, the `LLMService` is restricted to **1 worker thread** to prevent memory corruption.
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
