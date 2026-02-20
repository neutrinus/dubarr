# Spec 001: Video Dubbing Pipeline

## 1. Overview
The **Video Dubbing Pipeline** is the core business process of Dubarr. It transforms a source video into a dubbed version with synchronized audio in one or more target languages.

## 2. Process Flow
The pipeline executes sequentially in 8 stages (plus initialization). Each stage is checkpointed in the database to allow resumption.

### 2.0 Initialization (Startup)
The system attempts to pre-load critical models (Whisper, Diarization, LLM, TTS) into VRAM on application startup to reduce latency. If VRAM is constrained, the `GPUManager` handles dynamic loading/unloading.

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
    1.  **Transcription:** Whisper converts speech to text with timestamps.
    2.  **Diarization:** Pyannote (or similar) identifies speaker segments (Speaker A, Speaker B).
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

### 2.5 Stage 5: Production (Per Language)
*   **Loop:** For each `target_lang` in request:
    1.  **Draft Translation:** LLM translates the script, respecting speaker style.
    2.  **Sync Loop:** (See Spec 002 for details).
        -   TTS generates audio.
        -   If audio duration > video duration, LLM shortens text.
        -   Repeats until synchronized.
    3.  **Mastering:** Applies EQ, compression, and panning based on speaker position (if detected).
*   **Output:**
    *   `final_{lang}_{index}.wav` (Individual segments).

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
