# Spec 001: Video Dubbing Pipeline

## 1. Overview
The **Video Dubbing Pipeline** is the core business process of Dubarr. It transforms a source video into a dubbed version with synchronized audio in one or more target languages.

## 2. Process Flow
The pipeline executes sequentially in 8 stages (plus initialization). Each stage is checkpointed in the database to allow resumption.
Each task works in its own folder, `TEMP_DIR/task_<id>` (`standalone` without a task): file names inside are fixed,
and checkpoints point at them, so a shared folder would let a retried task pick up another task's audio.

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
*   **Input:** `vocals.wav` (diarization, voice references) and the original mix (transcription).
*   **Action:**
    1.  **Diarization:** Pyannote identifies speaker segments. Model is unloaded immediately after.
    2.  **Transcription:** Whisper converts speech to text. Model is loaded on-demand and unloaded immediately after.
        Whisper hears the original mix (`ASR_SOURCE=original`, default; `vocals` uses the Demucs stem): it is
        trained on noisy audio, while separation can cut or distort speech under loud effects. Diarization and voice
        references keep using the separated vocals. The first pass uses VAD and word timestamps. VAD also drops speech under loud noise (commentary over a fight,
        a referee's countdown), so a second pass (`ASR_GAP_PASS`, on) transcribes every gap of at least `ASR_GAP_MIN`
        (2 s) without VAD and keeps only confident segments: `no_speech_prob < 0.5`, `avg_logprob >= -0.8`,
        compression ratio <= 2.4, mean word probability >= 0.5, and not a known hallucination ("Thank you.",
        "Instruction.").
*   **Output:**
    *   `diarization.json` (Time ranges mapped to speakers, plus each speaker's centroid embedding for the voice
        bank, Spec 006, 4).
    *   `transcription.json` (Text segments with timestamps).

#### 2.2a Dialogue units (script creation)
Whisper splits speech into segments that often end mid-sentence. Translating such fragments one by one loses or
duplicates content, because the target language orders the sentence differently. The script is therefore built from
**dialogue units**:
1.  Each Whisper segment gets the speaker with the largest overlap. A segment whose length is far beyond what its words
    can take (e.g. one word stretched over 25 s of music) is clamped to `max(2 s, 1 s + syllables / 3)`.
2.  Consecutive segments of the same speaker are joined when the pause between them is under `MERGE_MAX_GAP` (0.3 s) and
    the unit stays under `MERGE_MAX_DURATION` (10 s), **or** when the earlier one does not end a sentence (no `.`, `!`,
    `?`, `…`), the pause is under `SENTENCE_MAX_GAP` (1.5 s) and the unit stays under `SENTENCE_MAX_DURATION` (20 s).
3.  Lines shorter than `MIN_SEGMENT_DURATION` are joined with a same-speaker neighbour.
4.  Each unit has a **time budget**: `end - start` is the time the original speaks (the target), and `max_end` adds the
    silence after it, up to `SLACK_AFTER_MAX` (1.5 s) and minus a 0.15 s guard before the next speech; `min_start`
    allows up to `SLACK_BEFORE_MAX` (0.5 s) of the silence before it, minus the same guard. The hard limit is
    `max_end - min_start`. The silence after is used first; a take starts early only when it would still run past
    `max_end`, and never before the previous dubbed line has ended.

### 2.2b Stage 2b: Media Metadata
*   **Input:** Path of the video.
*   **Action:**
    1.  Parses the path (Sonarr/Radarr naming such as `Show/Season 2/Show - S02E02 - Title WEBDL-1080p.mkv`, scene names,
        `{tvdb-…}`/`{imdb-…}`/`{tmdb-…}` ids, `Movie (2010)`) into title, year, season and episode, and the episode
        title that follows `SxxEyy - ` up to the quality tag (`WEBDL-480p`).
    2.  Reads Kodi-style `.nfo` files next to the video (`<video>.nfo`, `tvshow.nfo`, `movie.nfo`) when present.
    3.  Fills the gaps online (`METADATA_LOOKUP=on`): TVmaze for episodes (no account; show synopsis, episode title and
        synopsis, main and guest cast with gender), TMDB for movies (only with `TMDB_API_KEY`).
    4.  Any failure is logged and the job continues; at worst the title guessed from the file name is used.
*   **Output:** `media_info.json` (cached step result; written to the debug folder in debug mode).

### 2.3 Stage 3: Global Context Analysis
*   **Input:** Transcription script, reference subtitles, media metadata.
*   **Action:**
    1.  LLM analyzes the full script to understand context, tone, and relationships between speakers, using the
        media metadata (synopsis, cast list) to name speakers after characters and to tell their gender. Without a cast
        entry the gender is a best guess from names, pronouns, forms of address and role; "unknown" only without any
        hint (it leaves the speaker with a generic designed voice and ungendered translation forms).
    2.  Generates a glossary for translation: the names that keep their original sound, with a phonetic spelling in
        the target language for the TTS. In a series the stored spellings are fixed, and every new spelling must pass
        a reading test before it is used (Spec 006, 3). The names in the episode title and the cast are given to the
        analysis and to the transcript corrector as names that occur in the episode.
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
    1.  Scores every 2-20 s line of each speaker: ASR confidence, speaking rate, length (6-14 s best), position, and
        **clarity**: the level of the Demucs vocals over the level of the rest (`no_vocals`) in the same span, in dB.
        A line under `REF_MIN_CLARITY_DB` (3 dB, crowd or engines as loud as the voice) loses most of its score; above
        12 dB it gains. OmniVoice clones the clip's character, so noise and screaming in a reference come back in the dub.
        **Calm score:** the loudness of the line's vocals is compared with the median of all lines of the same speaker
        over the whole episode (calm talk can come before or after the shouting). A line 4 dB or more above that median
        is shouted and loses 5 points per dB above 3 dB, up to 40; a line within 2 dB of it gains 5. A speaker is only
        compared with itself, so a loud voice is not penalised for being loud.
    2.  **Short clips grow:** a candidate under 6 s is joined with the same speaker's lines right before or after it
        (pauses of at most 0.6 s), as long as each joined line is clean enough (clarity at least `REF_MIN_CLARITY_DB`)
        and the clip stays within 10 s; OmniVoice clones best from 6-10 s. The transcripts are joined too.
    3.  Tests the best candidates (up to 8) for Zero Crossing Rate (ZCR <= 0.15, after denoising) and keeps the first
        `REF_AUDITION_CANDIDATES` (3) that pass, best score first. When none passes, the tested candidate with the
        lowest ZCR is used: the speaker's own voice beats another speaker's reference. Which of the kept clips the TTS
        clones best is decided by the voice audition in Stage 5 (Spec 002, 2.3).
*   **Output:**
    *   `refs/{speaker_id}.wav`, `refs/{speaker_id}_2.wav`, ... (reference candidates for TTS, best score first).

### 2.5 Stage 5: Production (Global Task Pool)
*   **Action:** All segments from all `target_langs` are combined into a single task pool to maximize resource utilization.
*   **Workflow:**
    1.  **Draft Translation:** LLM generates drafts for all languages (see 2.5a).
    1b. **Voice audition** (`TAKE_CHECK` `on` or `audition`): each speaker's reference candidates are tried on one of
        the speaker's own draft lines and ranked by how intelligible the take is (Spec 002, 2.3).
    2.  **Global Task Queue:** A unified list of all segments is created.
    3.  **Concurrent Execution:** A global Thread Pool of orchestrators processes segments.
    4.  **Sync Loop:** Each segment uses `LLMService` and `TTSService` via RPC.
*   **Resource Management:** GPU-heavy tasks are prioritized (Refinement > Drafting) across all languages simultaneously.
*   **Output:** `final_{lang}_{index}.wav` segments, sorted and grouped by language.
*   **Production report:** after the sync loop, one log line per language summarizes it (lines, slot fill, statuses,
    take-check similarity). With `DEBUG=1`, `report_{lang}.tsv` in the debug folder lists every line: time, speaker,
    slot and limit, take duration and fill, status, voice reference type, judge score, voice match, text match,
    similarity, attempts, source text, dub text, what the take check's Whisper heard (`heard`) and what the judge's CTC
    model read (`ctc_heard`), side by side to calibrate the judge against Whisper.

#### 2.5a Draft translation
The goal is a complete translation that fits the timing. Run time is secondary.
1.  **Syllable budget per line.** The translator gets, per line, `min_syllables`, `target_syllables` and `max_syllables`,
    derived from the time budget and `DUB_SYLLABLES_PER_SECOND` (default 5.0, the pace of the synthesized voice):
    target = speech time × rate, max = (`max_end - min_start`) × rate, min = 75% of the target. LLMs keep syllable counts much
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
5.  **Length variants.** Besides the main translation (aimed at `target_syllables`), the translator returns a `short`
    variant (towards `min_syllables`, essentials only) and a `long` one (towards `max_syllables`, fuller phrasing, no
    invented facts). All carry the same information as far as their length allows. The sync loop picks between them
    before synthesis (Spec 002, 2.2). A line fixed by the review keeps only the fixed text.
6.  Lines that are pure sound markers (`[music]`, `(laughter)`) are not dubbed.

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
        4.  Deletes the task's folder (unless `debug_mode` is enabled). A failed task keeps it, so a retry reuses the
            checkpointed stages.
        5.  Updates task status to `DONE`.
    
    ## 3. Data Persistence
-   **Task Table:** Tracks overall progress (`QUEUED`, `PROCESSING`, `DONE`, `ERROR`).
-   **Job Steps Table:** Stores the result of each stage (`result_data` JSON) to enable skipping completed steps on retry.
