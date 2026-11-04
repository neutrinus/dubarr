# Dubarr Architecture

## 1. System Overview

**Dubarr** is an automated video dubbing pipeline. It takes a video file, separates the audio, transcribes it,
translates the content while preserving speaker identity and timing, and adds dubbed audio tracks to the same file in
place. All models run locally.

The system is designed as a **Modular Monolith** with a clear separation between the API (Entry Point), Core Domain Logic (Pipeline), and Infrastructure (I/O, External Services).

## 2. Architectural Layers

### 2.1. API Layer (`src/api`)
- **Responsibility:** Handles HTTP requests, file uploads, and status reporting.
- **Technology:** FastAPI, Jinja2 (Templates).
- **Key Components:**
    - `server.py`: Main application entry point.
    - **Authentication:** Basic Auth for protected routes.
    - **Webhooks:** Accepts external notifications for file processing.

### 2.2. Core Layer (`src/core`)
- **Responsibility:** Contains all business logic, the dubbing pipeline, and orchestration.
- **Key Components:**
    - `pipeline.py`: The central workflow engine (stages, step caching, Stage 5 pool, mix, mux).
    - `worker.py`: Background worker that picks tasks from the database queue.
    - `gpu_services.py`: Resource-bound services (LLM, TTS) that serialize GPU access via priority queues (internal RPC).
    - `gpu_manager.py`: Picks the CUDA device with the most free VRAM.
    - `synchronizer.py`: The sync loop: makes each translated line fit its time budget (Spec 002).
    - `take_judge.py`: Scores takes (signal artifacts, voiceprint match, CTC text match) without Whisper (Spec 002).
    - `voice_audition.py`: Ranks each speaker's voice references by the judge's score of their clones (Spec 002).
    - `series_memory.py`: Glossary and voice bank shared by the episodes of a series (Spec 006).
    - `media_context.py`: Identifies the show/episode or movie and gathers metadata for the prompts (Spec 001).
    - `subtitles.py`: Sidecar and embedded subtitles: cues by language, alignment to lines, SDH names (Spec 001).
    - `script_check.py`: Stage 2c script verification, energy classes, shout/calm merge (Spec 001).
    - `whisper_engine.py`, `diarization_engine.py`, `audio.py`: ASR, speaker diarization, Demucs and ffmpeg mixing.
    - `llm_engine.py`: Analysis, transcript correction, translation and refinement.
    - `tts_manager.py`: Voice reference selection and Text-to-Speech calls.
    - `report.py`: Per-line production report.

### 2.3. Infrastructure Layer (`src/infrastructure`)
- **Responsibility:** Handles all external interactions (Database, File System, External APIs).
- **Key Components:**
    - `database.py`: SQLite adapter for task management and step caching.
    - `ffmpeg.py`: Wrapper for FFmpeg operations (media processing).
    - `tts_client.py`: Starts, restarts and calls the OmniVoice server subprocess.
    - `llm_client.py`: OpenAI-compatible client for an external LLM server (`LLM_API_URL`).
    - `metadata_client.py`: `.nfo` parser, TVmaze and TMDB clients.
    - `monitor.py`: Resource usage monitoring.

### 2.4. Internal Microservices
-   **TTS Server (`src/omnivoice_server.py`):**
    -   **Responsibility:** Hosts the OmniVoice speech synthesis engine.
    -   **Architecture:** Runs as a separate Flask process on `localhost:5051`, on the app's own interpreter.
    -   **Reason:** A separate process can be restarted after a CUDA error without killing the worker.
    -   **Lifecycle:** The main application (`src/infrastructure/tts_client.py`) manages the startup and shutdown of this subprocess.

## 3. Data Flow

1.  **Ingestion:** User uploads a video or sends a webhook. A `Task` is created in the SQLite database with status `QUEUED`.
2.  **Processing:** The `JobWorker` picks up the task and initializes `DubbingPipeline`.
3.  **Pipeline Stages:**
    -   **Stage 1: Separation:** Vocals are isolated from background music/noise (Demucs).
    -   **Stage 2: Analysis:** Diarization identifies speakers, Whisper transcribes the original mix; the script is built
        from dialogue units with time budgets. **Stage 2b:** media metadata (show, episode, cast).
        **Stage 2c:** script verification (unsure lines heard again, speaker check by voiceprint).
    -   **Stage 3: Global analysis:** LLM summary, glossary, speaker profiles, transcript correction.
    -   **Stage 4: Voice references:** reference clips per speaker are scored and ZCR-checked.
    -   **Stage 5: Production (Global Task Pool):**
        -   Draft translations per language, designed voices and the voice audition.
        -   All segments from all target languages are unified into a single task pool, processed by 4 orchestrator
            threads regardless of language.
        -   Resources (LLM, TTS) are accessed via the internal RPC layer with priority queues.
        -   **Sync Loop:** each line is synthesized, judged and refined until it fits its time budget.
    -   **Stage 6: Mixing:** The dub is mixed over the original audio with sidechain ducking.
    -   **Stage 7: Muxing:** The dubbed tracks are added to the file in place (original kept as `.bak`); the series
        memory is updated.
4.  **Completion:** The task status is updated to `DONE`.

## 4. Key Technical Decisions

-   **Single Python Environment:** The Docker image has one Python 3.12 venv (`.venv`) for the app, Whisper, Pyannote,
    Demucs, llama-cpp and OmniVoice.
-   **Step Caching:** The pipeline checks `job_steps` in the DB before running a stage. If a result exists, it's skipped. This allows resuming failed jobs without re-processing expensive steps.
-   **Internal RPC (GPU Services):** Resource-heavy tasks (LLM, TTS) are offloaded to dedicated service threads with priority queues.
    -   **LLM Stability:** Due to thread-safety limitations in `llama-cpp-python` (CUDA), the `LLMService` is restricted to **1 worker thread** to prevent Segmentation Faults.
    -   **Concurrency:** The main pipeline uses multiple orchestrator threads (default: 4) to handle logic, while the RPC layer ensures serialized access to GPU resources.
-   **FFmpeg Direct Usage:** `subprocess` calls FFmpeg directly for complex filter chains (side-chain ducking, speed
    adjustment); simple operations live in `FFmpegWrapper`.
-   **Synchronous & Asynchronous Mix:** The web server is async (FastAPI), but the pipeline runs in a separate thread/process (Worker) due to heavy CPU/GPU blocking operations.
-   **Local Database:** SQLite is used for simplicity and portability. It stores both task metadata and JSON blobs of intermediate results.

## 5. Development Guidelines

-   **Do not** put business logic in `src/api/server.py`.
-   **Do not** call external APIs directly in `src/core` without an abstraction (Manager/Engine).
-   **Always** respect the `abort_event` in long-running loops to allow graceful shutdown.
-   **Debug Mode:** With `DEBUG=1`, intermediate files (segments, drafts, `report_<lang>.tsv`) are kept in
    `videos/debug_{filename}/`.
