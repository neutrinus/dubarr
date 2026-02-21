# Spec 002: Audio Synchronization Engine

## 1. Goal
The **Audio Synchronization Engine** ensures that the generated translated speech matches the duration and lip movements of the original video segment as closely as possible. It is implemented in `src/core/synchronizer.py`.

## 2. Core Logic: The Sync Loop

The process iterates for each dialogue segment until an acceptable duration is achieved or the attempt limit is reached.

### 2.1 Inputs
-   **Target Duration:** The time available for the segment in the original video.
-   **Current Text:** The translated text (starts with the draft).
-   **Speaker Profile:** Reference audio and style settings.
-   **Full Script Context:** Surrounding segments for grammatical continuity.
-   **Attempt Limit:** Default is 5 attempts.

### 2.2 Execution Steps

1.  **Sanitization:**
    -   Before synthesis, text is sanitized to remove or replace problematic characters (e.g., `…` -> `...`, `„` -> `"`) that could trigger internal TTS engine errors.

2.  **Synthesize (TTS):**
    -   Generate audio using the `TTSService` for the `current_text`.
    -   Measure the actual duration of the generated audio (`actual_dur`).

3.  **Measure Deviation (Delta):**
    -   Calculate `delta = actual_dur - target_dur`.

4.  **Acceptance Check:**
    -   **Strict Criteria:** `actual_dur <= (target_dur + 0.05s)`.
    -   **If Accepted:** Proceed to Step 6.
    -   **If Rejected:** Proceed to Step 5.

5.  **Refinement (LLM):**
    -   **Context Awareness:** The LLM receives the **previous** and **next** lines of the script to ensure the new version maintains grammatical consistency and natural flow.
    -   **Prompt:** Ask the LLM to rewrite `current_text` to match the target duration while strictly adhering to the target language's grammar rules.
    -   **Priority:** Refinement tasks are sent to `LLMService` with **Priority 1** (High).
    -   **Update:** Set `current_text` to the new version.
    -   **Repeat:** Go back to Step 2 (up to 5 times).

6.  **Finalization:**
    -   **Success:** Return the accepted audio file path.
    -   **Fallback (Max Attempts Reached):** Return the *longest* attempt that satisfies the strict duration criteria. If none fit, return the shortest available attempt with a warning.
    -   **Critical Failure Fallback:** If synthesis fails completely for a segment, the engine returns the original audio segment (if available) or a silent filler to prevent the entire pipeline from stalling.

### 2.3 Voice Reference Strategy
To ensure consistent voice cloning, the engine uses a hierarchical strategy to select the reference audio for each segment:
1.  **Dynamic Extraction:** Attempts to extract a clean sample from the *current* original audio segment (if long/clean enough). This preserves local emotion/prosody.
2.  **Last Good Sample:** If (1) fails, uses the last successfully synthesized segment's reference for this speaker.
3.  **Golden Reference:** If (1) and (2) fail, falls back to the best global sample found in Stage 4.

### 2.4 Post-Processing (Speed Adjustment)
Even after the loop, minor discrepancies may exist. The engine applies a final FFmpeg `atempo` filter:
-   **Stretch/Shrink:** Adjusts playback speed by up to ±20% (0.8x to 1.2x).
-   **Pitch Correction:** Ensures pitch remains natural despite speed changes.

## 3. Configuration
-   **`attempt_limit`**: Configurable (default: 5).
-   **`zcr_threshold`**: Zero Crossing Rate threshold for validating audio quality (default: 0.15).
-   **`temp_dir`**: Directory for storing intermediate WAV files.
