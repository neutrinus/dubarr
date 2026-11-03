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
    1.  Extracts original stereo audio track.
    2.  Uses an ML model (e.g., Demucs via `prep_audio`) to separate vocals from background music/noise.
*   **Output:**
    *   `orig.wav` (Original stereo audio)
    *   `vocals.mp3` (Clean speech separated by Demucs)

### 2.2 Stage 2: Audio Analysis
*   **Input:** `vocals.wav`.
*   **Action:**
    1.  **Diarization:** Pyannote identifies speaker segments. Model is unloaded immediately after.
    2.  **Transcription:** Whisper converts speech to text. Model is loaded on-demand and unloaded immediately after.
*   **Output:**
    *   `diarization.json` (Time ranges mapped to speakers).
    *   `transcription.json` (Text segments with timestamps).

#### 2.2a Dialogue units (script creation)
Whisper splits speech into segments that often end mid-sentence. Translating such fragments one by one loses or
duplicates content, because the target language orders the sentence differently. The script is therefore built from
**dialogue units**:
1.  Each Whisper segment gets the speaker with the largest overlap. A segment whose length is far beyond what its words
    can take (e.g. one word stretched over 25 s of music) is clamped to `max(2 s, 1 s + syllables / 1.5)`.
2.  Consecutive segments of the same speaker are joined when the pause between them is under `MERGE_MAX_GAP` (0.3 s) and
    the unit stays under `MERGE_MAX_DURATION` (10 s), **or** when the earlier one does not end a sentence (no `.`, `!`,
    `?`, `…`), the pause is under `SENTENCE_MAX_GAP` (1.5 s) and the unit stays under `SENTENCE_MAX_DURATION` (20 s).
3.  Lines shorter than `MIN_SEGMENT_DURATION` are joined with a same-speaker neighbour.
4.  Each unit has a **time budget**: `end - start` is the time the original speaks (the target), and `max_end` adds the
    silence after it, up to `SLACK_AFTER_MAX` (1.5 s) and minus a 0.15 s guard before the next speech (the hard limit).
    The silence before a unit is never used.

### 2.2b Stage 2b: Media Metadata
*   **Input:** Path of the video.
*   **Action:**
    1.  Parses the path (Sonarr/Radarr naming such as `Show/Season 2/Show - S02E02 - Title WEBDL-1080p.mkv`, scene names,
        `{tvdb-…}`/`{imdb-…}`/`{tmdb-…}` ids, `Movie (2010)`) into title, year, season and episode.
    2.  Reads Kodi-style `.nfo` files next to the video (`<video>.nfo`, `tvshow.nfo`, `movie.nfo`) when present.
    3.  Fills the gaps online (`METADATA_LOOKUP=on`): TVmaze for episodes (no account; show synopsis, episode title and
        synopsis, main and guest cast with gender), TMDB for movies (only with `TMDB_API_KEY`).
    4.  Any failure is logged and the job continues; at worst the title guessed from the file name is used.
*   **Output:** `media_info.json` (cached step result; written to the debug folder in debug mode).

### 2.3 Stage 3: Global Context Analysis
*   **Input:** Transcription script, reference subtitles, media metadata.
*   **Action:**
    1.  LLM analyzes the full script to understand context, tone, and relationships between speakers, using the
        media metadata (synopsis, cast list) to name speakers after characters and to tell their gender.
    2.  Generates a glossary or style guide for translation (including the character names).
    3.  The translator gets the show/episode, the synopsis and each line's speaker gender, so gendered forms
        (e.g. Polish past tense) match the speaker; the per-line duration refinement gets the speaker and title too.
*   **Reference subtitles:** `<video>.srt`, else `<video>.<lang>[.hi|.sdh].srt` in a non-target language (English and full
    subtitles preferred, forced subtitles skipped), else the first embedded subtitle stream.
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
    1.  **Draft Translation:** LLM generates drafts for all languages (see 2.5a).
    2.  **Global Task Queue:** A unified list of all segments is created.
    3.  **Concurrent Execution:** A global Thread Pool of orchestrators processes segments.
    4.  **Sync Loop:** Each segment uses `LLMService` and `TTSService` via RPC.
*   **Resource Management:** GPU-heavy tasks are prioritized (Refinement > Drafting) across all languages simultaneously.
*   **Output:** `final_{lang}_{index}.wav` segments, sorted and grouped by language.

#### 2.5a Draft translation
The goal is a complete translation that fits the timing. Run time is secondary.
1.  **Syllable budget per line.** The translator gets, per line, `min_syllables`, `target_syllables` and `max_syllables`,
    derived from the time budget and `DUB_SYLLABLES_PER_SECOND` (default 5.0, the pace of the synthesized voice):
    target = speech time × rate, max = time to `max_end` × rate, min = 75% of the target. LLMs keep syllable counts much
    better than seconds.
2.  **Windows with context.** Lines are translated in windows of `TRANSLATION_WINDOW` lines (default 20). Each request
    also shows the previous 6 lines with their final translations and the next 4 source lines, read-only, so the
    translation reads as one text and terms, forms of address and gender stay consistent.
3.  **Information is moved, not dropped.** The translator must keep every piece of information of the window: names,
    numbers, results, reasons, who does what. Fillers ("you know", "well", repetitions) go first. When a line cannot
    hold everything within its `max_syllables`, the information may move to a neighbouring line of the same speaker that
    has spare budget. It must never move to another speaker.
4.  **Completeness review** (`TRANSLATION_REVIEW=on`, default). A second LLM pass compares each window's source and
    translation, lists the information that is missing or wrong and returns corrected lines within their budgets. Only
    the corrected lines are replaced.
5.  Lines that are pure sound markers (`[music]`, `(laughter)`) are not dubbed.

### 2.6 Stage 6: Final Mix
*   **Input:** `final_{lang}_{index}.wav` segments, `orig.wav`.
*   **Action:**
    1.  Merges all speech segments into a single track.
    2.  Mixes it over the **original** audio (not the Demucs accompaniment, which also removes part of the sound effects).
    3.  Sidechain ducking lowers the original while the dub speaks; a low-passed copy of the original ("ambient ghost")
        keeps the room sound.
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
        3.  Generates a detailed performance report logged to the system console.
        4.  Cleans up temporary files (unless `debug_mode` is enabled).
        5.  Updates task status to `DONE`.
    
    ## 3. Data Persistence
-   **Task Table:** Tracks overall progress (`QUEUED`, `PROCESSING`, `DONE`, `ERROR`).
-   **Job Steps Table:** Stores the result of each stage (`result_data` JSON) to enable skipping completed steps on retry.
