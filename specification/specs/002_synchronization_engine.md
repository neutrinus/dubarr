# Spec 002: Audio Synchronization Engine

## 1. Goal
The **Audio Synchronization Engine** makes the translated speech fit the timing of the original line **without losing
content** and without sounding rushed or dragged. It is implemented in `src/core/synchronizer.py`, with duration control
in the OmniVoice server (`src/omnivoice_server.py`).

Priorities, in order: (1) nothing of the content is lost, (2) the line is intelligible and sounds natural,
(3) it fits its time budget.

## 2. Core Logic: The Sync Loop

### 2.1 Inputs
-   **Time budget** (see Spec 001, 2.2a): the **target** is the time the original speaks (`end - start`); the **limit**
    is `max_end - min_start`: the silence after the line, plus a little silence before it (Spec 001, 2.2a).
-   **Current Text:** The translated text (starts with the draft).
-   **Speaker Profile:** Reference audio and style settings.
-   **Full Script Context:** Surrounding lines (source and translation) for grammatical continuity.
-   **Attempt Limit:** 3 text versions per line.

### 2.2 Execution Steps

1.  **Sanitization:** problematic characters (e.g. `…` -> `...`, `„` -> `"`) are replaced before synthesis.

2.  **Pick a length variant (no synthesis).** When the draft has variants, the OmniVoice server estimates the
    natural length of each (`/estimate`, the model's own text-based estimator; no audio is generated). Content comes
    first: a variant *fits* when duration control can bring it into the limit (estimate <= limit x `TTS_MAX_SPEEDUP`).
    When the main translation fits, the short variant is never picked up front; between the main translation and the
    long variant, the one closest to the target wins (running past the limit counts three times as much as falling
    short). When the main translation does not fit, the longest variant that fits is used, or the shortest if none
    does. If there are no variants or the estimate fails, the main translation is used.

3.  **Synthesize with duration control (OmniVoice).** The server estimates the natural length of the text and asks the
    model for a duration only when needed, within a natural-sounding range:
    -   natural length within `[TTS_FILL_RATIO × target, limit]` (default ratio 0.9): natural pace;
    -   shorter: slow down towards the target, but not below `TTS_MIN_SPEED` (default 0.88);
    -   longer than the limit: speed up to the limit, but not above `TTS_MAX_SPEEDUP` (default 1.2).
    Long lines are synthesized in sentence chunks; each chunk gets its share of the target and of the limit.

4.  **Broken-take check.** A take is broken when it is
    -   *swallowed*: far shorter than its text can be spoken (more than `MAX_PLAUSIBLE_SYLLABLES_PER_SECOND`, default
        9 syllables per second); the model produced mostly silence, which OmniVoice then trims; or
    -   *unintelligible* (`TAKE_CHECK`, on by default): Whisper transcribes the take in the target language (no VAD, no
        conditioning on previous text) and the letters-only similarity to its text is under `TAKE_MIN_SIMILARITY`
        (default 0.5). This catches noise, screams and gibberish cloned from a noisy voice reference. Lines under
        4 syllables ("Wow!") are not checked, the ASR spells interjections too freely.
    A broken take is synthesized again with the next voice reference (line clip -> golden -> last good), up to 2 extra
    syntheses. It is never accepted while a sound one exists; when every reference fails, the least broken take is
    kept (the most similar one; swallowed counts as worst). The check model runs on the CPU by default: on the TTS's
    GPU it took the VRAM OmniVoice needs for long takes and made it crash. `TAKE_CHECK_DEVICE=auto` puts it on a GPU
    only when one has 8.5 GB free (the model plus the TTS headroom). It is released after Stage 5.

5.  **Acceptance check.** Accepted when `actual <= limit × MAX_ACCEPT_SPEEDUP + 0.05 s` and
    `actual >= MIN_FILL_RATIO × target` (default 0.7). A short take is kept when the LLM finds nothing to add (it
    returns the same text): short original lines like "Wow!" stay short.

6.  **Unused variant first.** A rejected take is retried with the unused variant whose estimate moves towards the
    target (shorter when too long, longer when too short), before asking the LLM.

7.  **Refinement (LLM)**, Priority 1 in `LLMService`, with the previous and next lines (source and translation):
    -   **Too long:** shorten by removing fillers and using shorter synonyms; names, numbers, results and reasons stay.
    -   **Too short:** restore details of the source that the draft left out, or phrase it more fully. Never invent facts.
    The loop goes back to step 3. It stops early when the LLM returns the same text.

8.  **Finalization:**
    -   **Success:** the accepted take.
    -   **Fallback (attempts exhausted):** among sound takes, the one that fits the limit and is closest to the target;
        if none fits, the shortest sound one. Swallowed takes are used only if nothing else exists.
    -   **Critical failure:** silence of the target length, so the pipeline does not stall (the original stays audible
        under it because the mix keeps the original audio).

### 2.3 Voice Reference Strategy
1.  **Line clip:** a clean clip of the line itself when it is 3-10 s long (carries the scene's emotion). Clean means
    ZCR < 0.25 and a clarity (vocals over the rest, see Spec 001, 2.4) of at least `REF_MIN_CLARITY_DB`; a line shouted
    over a cheering crowd uses the golden reference instead.
2.  **Golden Reference:** the best scored sample of the speaker from Stage 4.
3.  **Last Good Sample:** the last clean line clip of this speaker.
4.  **Global fallback:** any golden or last good sample.
A retry after a swallowed take skips the references already tried for that line.

### 2.4 Post-Processing (Speed Adjustment)
A take that would run past `max_end` starts earlier in the mix, by its overrun but not before `min_start` and never
before the previous dubbed line has ended. The final mix speeds a take up (pitch-preserving `atempo`) when it still
overruns, by at most 1.2x. Takes are
never stretched in the mix; slowing down is done by OmniVoice, which sounds natural.

## 3. Configuration
-   `DUB_SYLLABLES_PER_SECOND` (5.0): pace used for the translator's syllable budgets.
-   `TTS_FILL_RATIO` (0.9), `TTS_MIN_SPEED` (0.88), `TTS_MAX_SPEEDUP` (1.2): OmniVoice duration control.
-   `MIN_FILL_RATIO` (0.7): below this share of the target a take is refined as too short.
-   `MAX_ACCEPT_SPEEDUP` (1.08): allowed overrun of the limit, sped up in the mix.
-   `MAX_PLAUSIBLE_SYLLABLES_PER_SECOND` (9): faster than this means a swallowed take.
-   `TAKE_CHECK` (on), `TAKE_MIN_SIMILARITY` (0.5), `TAKE_CHECK_DEVICE` (cpu): the intelligibility check of every take.
-   `zcr_threshold` (0.25): Zero Crossing Rate limit for line-clip references.
