# Spec 001: Video Dubbing Pipeline

## 1. Overview
The **Video Dubbing Pipeline** is the core business process of Dubarr. It transforms a source video into a dubbed version with synchronized audio in one or more target languages.

## 2. Process Flow
The pipeline executes sequentially in 8 stages (plus initialization). Each stage is checkpointed in the database to allow resumption.

### 2.0 Initialization (Startup)
The system verifies model presence and downloads missing files. To maximize available VRAM, models follow a **Lazy Loading** strategy: they are loaded into memory only when needed and unloaded immediately after their stage completes.

### Stage 1: Audio Separation
*   **Input:** Original video file path.
*   **Action:**
    1.  Extracts audio track.
    2.  Uses an ML model (e.g., Demucs/Spleeter via `prep_audio`) to separate vocals from background music/noise.
*   **Output:**
    *   `vocals.wav` (Clean speech)
    *   `accompaniment.wav` (Music/SFX)

### 2.2 Stage 2: Audio Analysis
*   **Input:** `vocals.wav`.
*   **Action:**
    1.  **Diarization:** Pyannote identifies speaker segments. Model is unloaded immediately after.
    2.  **Transcription:** Whisper converts speech to text. Model is loaded on-demand and unloaded immediately after.
*   **Output:**
    *   `diarization.json` (Time ranges mapped to speakers).
    *   `transcription.json` (Text segments with timestamps).

### 2.3 Stage 3: Global Context Analysis
*   **Input:** Transcription script.
*   **Action:**
    1.  LLM analyzes the full script to understand context, tone, and relationships between speakers.
    2.  Generates a glossary or style guide for translation.
*   **Output:**
    *   `context.json` (Global context for translation).
    *   `speakers.json` (Speaker profiles: age, gender, role).

### 2.4 Stage 4: Voice Reference Extraction
*   **Input:** `vocals.wav`, `transcription.json`, `diarization.json`.
*   **Action:**
    1.  Scans for clean audio segments for each speaker.
    2.  Filters out short/noisy clips using Zero Crossing Rate (ZCR) and signal-to-noise ratio checks.
    3.  Selects the best 3-5 second sample for voice cloning.
*   **Output:**
    *   `refs/{speaker_id}.wav` (Reference audio for TTS).

### 2.5 Stage 5: Production (Global Task Pool)
*   **Action:** All segments from all `target_langs` are combined into a single task pool to maximize resource utilization.
*   **Workflow:**
    1.  **Draft Translation:** LLM generates drafts for all languages (sequential batching).
    2.  **Global Task Queue:** A unified list of all segments is created.
    3.  **Concurrent Execution:** A global Thread Pool of orchestrators processes segments.
    4.  **Sync Loop:** Each segment uses `LLMService` and `TTSService` via RPC.
*   **Resource Management:** GPU-heavy tasks are prioritized (Refinement > Drafting) across all languages simultaneously.
*   **Output:** `final_{lang}_{index}.wav` segments, sorted and grouped by language.

### 2.6 Stage 6: Final Mix
*   **Input:** `final_{lang}_{index}.wav` segments, `accompaniment.wav`.
*   **Action:**
    1.  Merges all speech segments into a single track.
    2.  Mixes with background music.
    3.  Applies "ducking" (lowering music volume during speech) if configured.
*   **Output:**
    *   `final_{lang}.ac3` (Mixed audio track).

### 2.7 Stage 7: Muxing
*   **Input:** Original video, `final_{lang}.ac3`.
*   **Action:**
    1.  FFmpeg replaces the original audio stream with the new dubbed track(s).
        2. Preserves video stream (copy codec) to avoid re-encoding quality loss.
    *   **Output:**
        *   `final_muxed.mkv` (Final video file).
    
    ### 2.8 Stage 8: Reporting & Cleanup
    *   **Action:**
        1.  Calculates total processing time and individual stage durations.
        2.  Calculates LLM performance metrics (tokens per second).
        3.  Logs a detailed performance report to the system logs.
        4.  Cleans up temporary files (unless `debug_mode` is enabled).
        5.  Updates task status to `DONE`.
    
    ## 3. Data Persistence
-   **Task Table:** Tracks overall progress (`QUEUED`, `PROCESSING`, `DONE`, `ERROR`).
-   **Job Steps Table:** Stores the result of each stage (`result_data` JSON) to enable skipping completed steps on retry.
