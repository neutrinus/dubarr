# Spec 002: Audio Synchronization Engine

## 1. Goal
The **Audio Synchronization Engine** ensures that the generated translated speech matches the duration and lip movements of the original video segment as closely as possible. It is implemented in `src/core/synchronizer.py`.

## 2. Core Logic: The Sync Loop

The process iterates for each dialogue segment until an acceptable duration is achieved or the attempt limit is reached.

### 2.1 Inputs
-   **Target Duration:** The time available for the segment in the original video.
-   **Current Text:** The translated text (starts with the draft).
-   **Speaker Profile:** Reference audio and style settings.
-   **Attempt Limit:** Default is 3 attempts.

### 2.2 Execution Steps

1.  **Synthesize (TTS):**
    -   Generate audio using the TTS engine for the `current_text`.
    -   Measure the actual duration of the generated audio (`actual_dur`).

2.  **Measure Deviation (Delta):**
    -   Calculate `delta = actual_dur - target_dur`.
    -   Calculate `abs_delta` (absolute difference).

3.  **Acceptance Check:**
    -   **Strict (Short Segments < 2s):** `abs_delta < 0.4s`.
    -   **Standard:** `abs_delta < 0.6s` OR `(abs_delta / target_dur) < 0.15` (15% tolerance).
    -   **If Accepted:** Proceed to Step 5.
    -   **If Rejected:** Proceed to Step 4.

4.  **Refinement (LLM):**
    -   **Prompt:** Ask the LLM to rewrite `current_text` to match the target duration.
        -   If `actual_dur > target_dur`: Request a shorter, more concise version.
        -   If `actual_dur < target_dur`: Request a slightly expanded version (if significantly short).
    -   **Update:** Set `current_text` to the new version.
    -   **Repeat:** Go back to Step 1.

### 2.3 Voice Reference Strategy
To ensure consistent voice cloning, the engine uses a hierarchical strategy to select the reference audio for each segment:
1.  **Dynamic Extraction:** Attempts to extract a clean sample from the *current* original audio segment (if long/clean enough). This preserves local emotion/prosody.
2.  **Last Good Sample:** If (1) fails, uses the last successfully synthesized segment's reference for this speaker.
3.  **Golden Reference:** If (1) and (2) fail, falls back to the best global sample found in Stage 4.

5.  **Finalization:**
    -   **Success:** Return the accepted audio file path and text.
    -   **Failure (Max Attempts Reached):** Return the *best* attempt (closest duration) with a warning status (`FALLBACK`).

### 2.3 Post-Processing (Speed Adjustment)
Even after the loop, minor discrepancies may exist. The engine applies a final FFmpeg `atempo` filter:
-   **Stretch/Shrink:** Adjusts playback speed by up to ±20% (0.8x to 1.2x).
-   **Pitch Correction:** Ensures pitch remains natural despite speed changes.

## 3. Configuration
-   **`attempt_limit`**: Configurable (default: 3).
-   **`zcr_threshold`**: Zero Crossing Rate threshold for validating audio quality (default: 0.15).
-   **`temp_dir`**: Directory for storing intermediate WAV files.
