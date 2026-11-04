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

4.  **Takes and the judge.** Each synthesis makes `TAKE_VERSIONS` (2) takes of the text at once, cloning the
    speaker's first two references in the order of 2.3. The **judge** scores each take in layers, cheapest first; a
    take that fails a layer is broken and the later layers skip it:
    1.  *missing or swallowed*: no usable audio, or far shorter than its text can be spoken (more than
        `MAX_PLAUSIBLE_SYLLABLES_PER_SECOND`, default 9 syllables per second): mostly silence, trimmed by OmniVoice.
    2.  *artifacts* (signal only, milliseconds): a squeak, i.e. frames with more than 35% of their energy above
        3.5 kHz and that band `JUDGE_HF_MAX_DB` (12 dB) above its usual level in the take, for 300 ms or more (the
        6:53 squeaks of the measured run: +29 dB for 1.08 s, the only such event in the episode); clipping (more than
        0.1% of samples at full scale); a silence over 1.2 s inside the take.
    3.  *voice match*: the take's voiceprint (Spec 006, 4.1; the audio goes to pyannote in memory, as in the
        diarization, because decoding a file needs torchcodec) against the print of the reference it cloned. Under
        `JUDGE_MIN_VOICE` (0.35) the clone is not that voice: the model drifted into another voice or into noise. A
        clone speaking another language is never close to its English clip: in S01E03 the accepted clones of the right
        person scored 0.41-0.59 (5th-75th percentile, median 0.56), so 0.5 rejected one clone in ten as a wrong voice.
    4.  *text match*: a CTC speech model of the target language (`JUDGE_ALIGN_MODEL`; default
        `jonatasgrosman/wav2vec2-large-xlsr-53-polish` for Polish, 300 M parameters, and `facebook/mms-1b-all` with
        the language's adapter for other languages; on the CPU) reads the take (greedy CTC decoding, no language
        model) and that reading is compared with the text on letters only, like the Whisper similarity (0..1). It has
        no fixed 30 s window, so it costs well under Whisper's 10 s per take. Under `JUDGE_MIN_TEXT` (0.5) the take is
        gibberish or lost words. Lines under 4 syllables skip this layer. Digits are left out of the text (the take
        speaks them as words). The likelihood of the text (`exp(-CTC loss / chars)`), used first, depends on how sure
        the model is of the voice rather than of the words: in the first run with it (S01E02) clear Polish from the
        cloned commentators scored 0.0-0.1, and 109 of 139 dropped lines were heard right by Whisper.
        The processor is loaded without the model's language model (it needs `pyctcdecode` and the score does not
        use it); when the language's model does not load, `facebook/mms-1b-all` is tried. When no CTC model loads,
        that is an error in the log, and in `whisper` mode the Whisper similarity takes the text match's place for
        the decision (same threshold): the first measured run had no text match, and 44 of 261 lines that Whisper
        heard as gibberish were accepted.
    A take's **judge score** is its text match (1.0 for lines that skip layer 4), halved when it failed the voice match
    and 0 when it was swallowed. The sound take with the best judge score goes on; when both versions are broken,
    one more synthesis is made with the next reference, up to 2 extra. A broken take is never accepted while a sound
    one exists; when every take is broken the least broken is kept (the best score; swallowed counts as worst).
    `TAKE_CHECK` modes: `judge` (default) as above, also for the voice audition; `whisper` runs the judge and, for
    measurement only, the old Whisper similarity next to it (both in the production report); `off` keeps only
    layers 1 and 2 (and no audition). The judge is cheap, so the earlier values `on` and `audition` read as `judge`. The judge's models are loaded on first use and released after Stage 5.

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
        if none fits, the shortest sound one. Broken takes are used only if nothing else exists.
    -   **Dropped:** when every take is broken and none is intelligible to `TAKE_DROP_SIMILARITY` (default 0.3), the
        line is not dubbed (status `DROPPED`): the original stays audible, which beats gibberish or screams in the dub.
        Only the words count here: the better of the take's text match and Whisper similarity, the judge score only
        when neither was measured (a swallowed or missing take counts as 0, a take with artifacts as its score). A
        take in a doubtful voice is still used: the English original showing through is worse.
    -   **Critical failure:** silence of the target length, so the pipeline does not stall (the original stays audible
        under it because the mix keeps the original audio).

### 2.3 Voice Reference Strategy
**Voice audition** (before the sync loop, `TAKE_CHECK` `on` or `audition`). Each speaker's reference candidates
(Spec 001, 2.4) synthesize the same test line: the speaker's own draft line (first target language) closest to
18 syllables, at natural pace, preferring a line without glossary respellings ("Ajś-łejw" is heard as "Ice Wave",
which lowers the score of every candidate alike). The judge scores each take as in 2.2 step 4 (its judge score). The candidates
are ranked by that score, best first. A speaker whose best score is under `AUDITION_LINE_REF_MIN` (default 0.8) does
not use line clips: a voice that clones badly from its cleanest clips (announcers shouting over the arena) clones
worse from an arbitrary line. Without the audition the Stage 4 order is kept and line clips are used.

**Designed voice** (`VOICE_DESIGN`, on by default). Before the audition, OmniVoice voice design synthesizes one clip
per speaker that has lines, from the attributes it understands: gender, age and pitch, taken from the speaker
profile (Stage 3; e.g. "male, middle-aged, low pitch"; age words match whole words only). With nothing known OmniVoice
picks a generic voice. The gender is measured, not guessed, when the speaker's reference clips tell: a median pitch
(autocorrelation over the voiced frames of up to `REF_AUDITION_CANDIDATES` clips) of at most 160 Hz is male, of
200-265 Hz female; between 160 and 200 Hz, and above 265 Hz, the profile's gender stays. A man shouting over the arena
speaks at 290-320 Hz (S01E03: the play-by-play man during the fights and the ring announcer, measured as women), so a
high pitch tells excitement, not a woman. The measurement overrules the analysis and the series memory (Spec 006),
which once stored the play-by-play man under the sideline reporter's name and gender, so his designed voice was a
woman's. When it contradicts the analysis, the analysis took the speaker for someone else, so its name is dropped
too: the translator saw "Jenny Taft (male)" and the review turned the man's lines into feminine forms. The clip speaks the speaker's own lines (about 30 syllables, not the audition test line unless
it is the speaker's only line) and joins the speaker's candidates as the last one; afterwards it is cloned like any
other reference, so the voice stays the same across the video. It is the fallback for speakers whose clips do not
clone: it is used only when none of the speaker's clips scores at least `AUDITION_LINE_REF_MIN` in the audition and
the designed voice scores better than the best of them. Its voice match compares OmniVoice with itself, so on scores
alone it beats every clone speaking another language (S01E03: 0.8 against 0.56 at the median), and in S01E03 it took
over seven speakers whose clips cloned well. Voice design was also trained on Chinese and English only and may be
unstable in other languages. It gives a voice to speakers without any clean clip.

A speaker only ever speaks with **its own voice** (its clips, line clips or designed voice): another speaker's voice in
its mouth confuses the viewer more than a generic voice does. And with **one voice**: the speaker's kind (the video's
clips, or the designed voice, as decided above) is used for every line, and the other kind never while the speaker has
a reference of its own kind. When every reference of that kind was tried for a line, the best of them is synthesized
again (a new take of the same voice). In S01E02 the takes of one speaker alternated between a clone and the designed
voice from line to line; in S01E03 the second take of each line still came from a clip, so 15-20% of the lines of a
speaker on the designed voice were clones. A speaker on the designed voice does not use line clips.

References are tried in this order; a retry after a broken take skips the ones already tried for that line:
1.  **Line clip:** a clean clip of the line itself when it is 3-10 s long (carries the scene's emotion). Clean means
    ZCR < 0.25 and a clarity (vocals over the rest, see Spec 001, 2.4) of at least `REF_MIN_CLARITY_DB`; a line shouted
    over a cheering crowd uses the speaker's references instead.
2.  **Speaker's references** of the speaker's voice kind that scored at least `TAKE_MIN_SIMILARITY` in the audition
    (all of them without an audition), in rank order.
3.  **Last Good Sample:** the last clean line clip of this speaker (not for a speaker in the designed voice).
4.  **Speaker's weak references** of that kind: those that scored under `TAKE_MIN_SIMILARITY`.
5.  **The other kind** (the designed voice for a cloned speaker, the clips for a designed one), by score.
When all of these were tried, the least broken take is kept (2.2 step 4). Only a speaker with no voice of its own at all
(no clip, design off or failed) uses another speaker's last good sample or reference, so the line is still dubbed.

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
-   `TAKE_CHECK` (judge | whisper | off), `TAKE_VERSIONS` (2): the judge of the voice audition and of every take, and
    how many takes each synthesis makes.
-   `JUDGE_HF_MAX_DB` (12), `JUDGE_MIN_VOICE` (0.35), `JUDGE_MIN_TEXT` (0.5), `JUDGE_ALIGN_MODEL`
    (per language, see 2.2 step 4), `JUDGE_DEVICE` (cpu): the judge's layers. Thresholds to be tuned on measured runs
    (`TAKE_CHECK=whisper`).
-   `TAKE_MIN_SIMILARITY` (0.7), `TAKE_CHECK_DEVICE` (cpu), `TAKE_CHECK_THREADS` (all cores): the Whisper similarity
    reported in `whisper` mode.
-   `TAKE_DROP_SIMILARITY` (0.3): a line whose every take is broken below this is not dubbed.
-   `VOICE_DESIGN` (on): one voice-design clip per speaker as an extra reference candidate.
-   `REF_AUDITION_CANDIDATES` (3), `AUDITION_LINE_REF_MIN` (0.8): reference candidates per speaker, and the audition
    score a speaker needs to use line clips.
-   `zcr_threshold` (0.25): Zero Crossing Rate limit for line-clip references.
