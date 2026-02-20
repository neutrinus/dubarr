# Dubarr Architecture

## 1. System Overview

**Dubarr** is an automated video dubbing and translation pipeline. It takes a video file as input, separates the audio, transcribes it, translates the content while preserving speaker identity and timing, and generates a new dubbed video file.

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
    - `pipeline.py`: The central workflow engine. Orchestrates the 7-stage process.
    - `worker.py`: Background worker that picks tasks from the database queue.
    - `synchronizer.py`: The "Sync Loop" engine. Ensures generated audio matches the video duration.
    - `diarization_engine.py`: Identifies speakers.
    - `llm_engine.py`: Interface for translation and text refinement.
    - `tts_manager.py`: Interface for Text-to-Speech generation.

### 2.3. Infrastructure Layer (`src/infrastructure`)
- **Responsibility:** Handles all external interactions (Database, File System, External APIs).
- **Key Components:**
    - `database.py`: SQLite adapter for task management and step caching.
    - `ffmpeg.py`: Wrapper for FFmpeg operations (media processing).
    - `monitor.py`: Resource usage monitoring.

## 3. Data Flow

1.  **Ingestion:** User uploads a video or sends a webhook. A `Task` is created in the SQLite database with status `QUEUED`.
2.  **Processing:** The `JobWorker` picks up the task and initializes `DubbingPipeline`.
3.  **Pipeline Stages:**
    -   **Stage 1: Separation:** Vocals are isolated from background music/noise.
    -   **Stage 2: Analysis:** Whisper transcribes audio; Diarization identifies speakers.
    -   **Stage 3: Context:** LLM analyzes the script for context and speaker traits.
    -   **Stage 4: Refinement:** Voice references are extracted and validated (ZCR check).
    -   **Stage 5: Production:**
        -   Text is translated draft-style.
        -   **Sync Loop:** Audio is generated. If duration mismatches, LLM rewrites text to fit constraints.
    -   **Stage 6: Mixing:** New vocals are mixed with original background audio.
    -   **Stage 7: Muxing:** Final audio is merged into the video container.
4.  **Completion:** The task status is updated to `DONE`.

## 4. Key Technical Decisions

-   **Step Caching:** The pipeline checks `job_steps` in the DB before running a stage. If a result exists, it's skipped. This allows resuming failed jobs without re-processing expensive steps.
-   **FFmpeg Direct Usage:** We use `subprocess` to call FFmpeg directly for complex filter chains (e.g., side-chain ducking, speed adjustment).
-   **Synchronous & Asynchronous Mix:** The web server is async (FastAPI), but the pipeline runs in a separate thread/process (Worker) due to heavy CPU/GPU blocking operations.
-   **Local Database:** SQLite is used for simplicity and portability. It stores both task metadata and JSON blobs of intermediate results.

## 5. Development Guidelines

-   **Do not** put business logic in `src/api/server.py`.
-   **Do not** call external APIs directly in `src/core` without an abstraction (Manager/Engine).
-   **Always** respect the `abort_event` in long-running loops to allow graceful shutdown.
-   **Debug Mode:** When `debug_mode` is true, intermediate files (segments, drafts) are preserved in `videos/debug_{filename}/`.
