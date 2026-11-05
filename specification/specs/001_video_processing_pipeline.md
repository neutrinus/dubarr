# Spec 001: Video Dubbing Pipeline

## 1. Overview
The **Video Dubbing Pipeline** is the core business process of Dubarr. It transforms a source video into a dubbed version with synchronized audio in one or more target languages.

## 2. Process Flow
The pipeline runs its stages in order (1, 2, 2b, 2c, 3, 4, 5, 6, 7, then reporting). Each stage's result is cached in the
database (`job_steps`, Spec 003) so a retried task resumes where it failed; Stage 5 also checkpoints per line.
Each task works in its own folder, `TEMP_DIR/task_<id>` (`standalone` without a task): file names inside are fixed,
and checkpoints point at them, so a shared folder would let a retried task pick up another task's audio.

### 2.0 Initialization (Startup)
The system verifies model presence and downloads missing files. To maximize available VRAM, models follow a **Lazy Loading** strategy: they are loaded into memory only when needed and unloaded immediately after their stage completes.

### 2.1 Stage 1: Audio Separation
*   **Input:** Original video file path.
*   **Action:**
    1.  Extracts original stereo audio track.
    2.  Demucs (`htdemucs_ft`, via `prep_audio`) separates vocals from background music and noise.
*   **Output:**
    *   `orig.wav` (Original stereo audio)
    *   `vocals.mp3` (Clean speech separated by Demucs)

### 2.2 Stage 2: Audio Analysis
*   **Input:** `vocals.mp3` (diarization, voice references) and the original mix (transcription).
*   **Action:**
    1.  **Diarization:** Pyannote identifies speaker segments. Model is unloaded immediately after.
    2.  **Transcription:** Whisper converts speech to text. Model is loaded on-demand and unloaded immediately after.
        Whisper hears the original mix (`ASR_SOURCE=original`, default; `vocals` uses the Demucs stem): it is
        trained on noisy audio, while separation can cut or distort speech under loud effects. Diarization and voice
        references keep using the separated vocals. The first pass uses VAD and word timestamps. VAD also drops speech under loud noise (commentary over a fight,
        a referee's countdown), so a second pass (`ASR_GAP_PASS`, on) transcribes every gap of at least `ASR_GAP_MIN`
        (2 s) without VAD. A pause of at least `ASR_GAP_MIN` between the words of one first-pass segment is a gap
        too: the segment is split there, and a word longer than 1.5 s counts only its last 1.5 s (the alignment
        stretches the first word after a noisy stretch over all of it). Over a fight in BattleBots S01E01 Whisper
        wrote one 144 s segment of eleven words, and the 2.5 minutes of commentary inside it were never heard. The
        gap pass keeps only confident segments: `no_speech_prob < 0.5`, `avg_logprob >= -0.8`,
        compression ratio <= 2.4, mean word probability >= 0.5, and not a known hallucination ("Thank you.",
        "Instruction.", "We'll be right back."). A gap-pass segment where nobody is heard speaking (diarized speech
        under 30% of it) and no reference subtitle cue lies is dropped as a hallucination over music: Star Trek
        S01E01 got "We'll be right back." 11 times, all without a speaker.
*   **Output:**
    *   `diarization.json` (Time ranges mapped to speakers, plus each speaker's centroid embedding for the voice
        bank, Spec 006, 4).
    *   `transcription.json` (Text segments with timestamps).

#### 2.2a Dialogue units (script creation)
Whisper splits speech into segments that often end mid-sentence. Translating such fragments one by one loses or
duplicates content, because the target language orders the sentence differently. The script is therefore built from
**dialogue units**:
1.  Each Whisper segment gets the speaker with the largest overlap. A segment whose length is far beyond what its words
    can take (e.g. one word stretched over 25 s of music) is clamped to `max(2 s, 1 s + syllables / 3)`, the window
    placed where the diarization hears the most speech in the segment, not at its start ("That's the heart of a
    galaxy" ran from 20:24 to 20:40 with the words at 20:39; dubbed from 20:24, the English line stayed audible).
2.  Consecutive segments of the same speaker are joined when the pause between them is under `MERGE_MAX_GAP` (0.3 s) and
    the unit stays under `MERGE_MAX_DURATION` (10 s), **or** when the earlier one does not end a sentence (no `.`, `!`,
    `?`, `…`), the pause is under `SENTENCE_MAX_GAP` (1.5 s) and the unit stays under `SENTENCE_MAX_DURATION` (20 s).
3.  Lines shorter than `MIN_SEGMENT_DURATION` are joined with a same-speaker neighbour.
4.  **Tiny slots.** A unit shorter than half the time its English words take (syllables / 5 per second; S01E03 timed
    "We'll be right back." at 0.12 s, and OmniVoice returned nothing) is lengthened to that time into the silence
    after it, then before it, never into another line.
5.  Each unit has a **time budget**: `end - start` is the time the original speaks (the target), and `max_end` adds the
    silence after it, up to `SLACK_AFTER_MAX` (1.5 s) and minus a 0.15 s guard before the next speech; `min_start`
    allows up to `SLACK_BEFORE_MAX` (0.5 s) of the silence before it, minus the same guard. The hard limit is
    `max_end - min_start`. The silence after is used first; a take starts early only when it would still run past
    `max_end`, and never before the previous dubbed line has ended.

### 2.2b Stage 2b: Media Metadata
*   **Input:** Path of the video.
*   **Action:**
    1.  Parses the path (Sonarr/Radarr naming such as `Show/Season 2/Show - S02E02 - Title WEBDL-1080p.mkv`, scene names,
        `{tvdb-…}`/`{imdb-…}`/`{tmdb-…}` ids, `Movie (2010)`) into title, year, season and episode, and the episode
        title that follows `SxxEyy - ` up to the quality tag (`WEBDL-480p`). Long-running shows have year seasons
        (`MythBusters - S2012E20 - Food Fables`): a 4-digit season of 19xx/20xx is a season, not a movie's year.
    2.  Reads Kodi-style `.nfo` files next to the video (`<video>.nfo`, `tvshow.nfo`, `movie.nfo`) when present.
    3.  Fills the gaps online (`METADATA_LOOKUP=on`): TVmaze for episodes (no account; show synopsis, episode title and
        synopsis, main and guest cast with gender), TMDB for movies (only with `TMDB_API_KEY`).
    4.  Any failure is logged and the job continues; at worst the title guessed from the file name is used.
*   **Output:** `media_info.json` (cached step result; written to the debug folder in debug mode).

### 2.2c Stage 2c: Script Verification
Whisper's mistakes on shouted commentary ("It's Robux! A fight in time!") used to reach the translator unmarked. This
stage gives every line a confidence and hears the unsure ones again (`ASR_VERIFY`, on).
1.  **Confidence.** A Whisper segment is unsure when its `avg_logprob` is under `ASR_UNSURE_LOGPROB` (-0.6) or one
    of its words has a probability under `ASR_UNSURE_WORD` (0.3).
2.  **Second hearing** of the unsure segments only, with the same Whisper model, loaded again after Stage 2b:
    1.  the segment (with 0.3 s around it) from the Demucs vocals: under a fight separation often helps, although it
        loses on the whole episode (Stage 2);
    2.  the segment from the original mix with an `initial_prompt` of the previous line and the names known to occur
        (episode title, cast, series glossary): robots and people are often misheard names;
    3.  the reference subtitle cues spoken during the segment, when there are subtitles.
    The candidates stay on the segment and go with it into its line (2.2a).
3.  **Who speaks.** Every Whisper segment of at least 1.5 s gets a voiceprint from the vocals (the voice bank's model,
    Spec 006, 4.1) and is compared with the diarization speakers' centroids. When it is at least
    `SPEAKER_RECHECK_MARGIN` (0.1) closer to another speaker than to the one it overlaps most, and that similarity is
    at least `SPEAKER_RECHECK_MIN` (0.5), the segment is that speaker's: a contestant's sentence inside the narrator's
    intro does not join the narrator's line.
4.  **The arbiter** is the transcript corrector of Stage 3 (2.3): an unsure line comes with its other hearings and
    subtitle, and the LLM picks one, combines them, or, when none makes sense, reconstructs the most likely sentence
    from the context (neighbouring lines, summary, names): it must sound like what Whisper heard and add no facts the
    context does not hold. Such a line is marked `guessed`; a corrected one `fixed`. Lines are never dropped for being
    unclear: the best version is dubbed. The corrector sees each line's speaker name and also fixes clear ASR errors
    of sure lines, as before. It may mark a line as a Whisper hallucination (a "Thank you." or "Instruction." written
    over music or noise, nothing like the rest of the scene): such a line is not dubbed and the original stays
    audible. It never moves a line to another speaker; the voiceprint check (3) decides that. It does report doubts
    about a speaker ("SPEAKER_03 is described as Jenny Taft, but line 57 addresses him as Chris"): only a signal,
    logged with a count per speaker and shown in the report's `speaker_doubt` column. The only thing it changes: the
    cast list no longer sets that speaker's gender (Spec 002, 2.3), since the name may belong to someone else.
    The second hearing leaves out known hallucinations ("We'll be right back.") like the gap pass does. When a line
    the corrector changed ends with words that the same speaker's next line (within `SENTENCE_MAX_GAP`) starts with,
    they are dropped from the next line: the second hearing of "...learning their answers to the game" heard into
    the next line, and "ABCs. ABCs." was dubbed. A next line that is nothing but the repetition stays.

### 2.3 Stage 3: Global Context Analysis
*   **Input:** Transcription script, reference subtitles, media metadata.
*   **Action:**
    1.  LLM analyzes the full script to understand context, tone, and relationships between speakers, using the
        media metadata (synopsis, cast list) to name speakers after characters and to tell their gender. Without a cast
        entry the gender is a best guess from names, pronouns, forms of address and role; "unknown" only without any
        hint (it leaves the speaker with a generic designed voice and ungendered translation forms).
    2.  Generates a glossary for translation: the names that keep their original sound, with a phonetic spelling in
        the target language for the TTS. Every name the episode is known to contain goes in, not only those the LLM
        notices: the cast, the episode title, and the capitalised names of the script and the reference subtitles
        (S01E03 had no entry for "Chuck Pitzer", and "przez Chucka Pitzera" came out as "szesczu kapicera"). An
        inflected Polish form keeps the respelled stem ("Czaka Pitzera"). In a series the stored spellings are fixed, and every new spelling must pass
        a reading test before it is used (Spec 006, 3). The names in the episode title and the cast are given to the
        analysis and to the transcript corrector as names that occur in the episode.
    3.  The translator gets the show/episode, the synopsis and each line's speaker gender, so gendered forms
        (e.g. Polish past tense) match the speaker; the per-line duration refinement gets the speaker and title too.
*   **Reference subtitles:** `<video>.srt`, else `<video>.<lang>[.hi|.sdh].srt` in a non-target language (English and full
    subtitles preferred, forced subtitles skipped), else the first embedded subtitle stream not tagged with a target
    language. Files are read as UTF-8, else as Windows-1250 (common for Polish subtitles).
    Cues that credit the subtitle makers are left out of every use (reference and target subtitles alike): a web
    address or domain ("www.addic7ed.com", "napisy24.pl"), a subtitle site's name, "Original air date", or the words
    they sign with followed by "by"/"and" ("Sync and corrections by", "Subtitles by") or a colon ("Tłumaczenie:",
    "Synchro:"). MythBusters S11E07 dubbed "Sync and corrections by atrn97g for www.addic7ed.com".
*   **Target-language subtitles:** `<video>.<lang>.srt` (ISO 639-1 or 639-2 code, e.g. `.pl.srt` / `.pol.srt`; full
    subtitles preferred over `.hi`/`.sdh`, forced subtitles skipped), else an embedded subtitle stream tagged with
    that language. They are a human translation, so they know the show's established names, terms, forms of address
    and the speakers' gendered forms better than an LLM. They are not read aloud: they are shortened for reading and
    timed differently from the dub. Each cue goes to the script line it overlaps most in time (at least a third of the
    cue's duration), so a line gets the cues spoken during it. Used by the draft translation (2.5a) and the series
    terms (Spec 006, 5).
*   **What else the subtitles tell:**
    0.  *Offset.* Subtitles of another release run early or late (MythBusters S2012E20: 1-2 s early). Every cue of 3+
        words that starts with the same words (up to 5) as a transcribed segment within 5 s gives one estimate; with
        at least 8 estimates and a median of at least 0.25 s, all cues are shifted by that median before anything
        below uses them.
    1.  *Missing speech.* A spoken cue of the reference subtitles (sound descriptions such as `[door opens]` and
        `♪` removed) of at least 1 s that the transcription does not cover (less than a third of the cue), and whose
        words are not in the transcription within 3 s of it (70% of the words; all of them for a cue of 1-2 words),
        is speech Whisper did not hear. Without the word check, half of MythBusters' "missing" cues were lines Whisper
        had heard a second later, and they were dubbed twice: it becomes a line of the script with the cue's text and times, and the speaker the
        diarization finds there. Only when the subtitles are timed to this audio: at least 70% of the spoken cues
        overlap the transcription (subtitles of another release are often shifted by seconds), and only where the
        diarization hears somebody speak for at least 30% of the cue: a credit or an on-screen sign is not speech.
    2.  *Speaker names.* Hearing-impaired subtitles (`.en.hi.srt`, `.en.sdh.srt`) label speakers (`KIRK:`,
        `[SPOCK]`). The label most of a speaker's labelled cues carry (at least 2 cues, 60%) is given to the
        analysis as a hint for naming that speaker.
    3.  *Gender.* In Polish target subtitles, the first-person past tense and conditional ("zrobiłem"/"zrobiłam",
        "zrobiłbym"/"zrobiłabym"; nouns in "-łem" such as "admirałem" not counted; cues with two speakers skipped)
        tell the speaker's gender: at least 3 forms, 80% of one gender. This overrules the analysis and the cast list
        (Spec 002, 2.3); it counts as a measured gender in the series memory.
    4.  *Forms of address.* The terms step (Spec 006, 5) also reads how speakers address each other in the subtitles
        (formal "pan kapitan", or "ty"); the translator keeps those forms.
*   **Output:**
    *   `context.json` (Global context for translation).
    *   `speakers.json` (Speaker profiles: age, gender, role).

### 2.4 Stage 4: Voice Reference Extraction
*   **Input:** `vocals.mp3`, the script (Stage 2a, corrected in Stage 3). The step is named "Stage 4: Transcription
    Correction" in the database for historical reasons; it does not change the text.
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
    3.  **Energy classes** (`EMOTION_REFS`, on). OmniVoice has no emotion control, but carries the energy of the clip
        it clones, so a speaker gets clips for each energy of its lines. Every line is measured on the vocals against
        the speaker's own medians: voice level (dB), pitch (semitones) and pace (syllables per second). Level and pitch
        add up to an arousal score: `shout` from +6 (e.g. +4 dB and +4 semitones), `lively` from +2.5, else `calm`;
        a pace 30% above the median adds 1. The candidates are ranked within each class (the calm score applies to the
        calm class only), and steps 2-4 pick up to `REF_AUDITION_CANDIDATES` clips per class. The clarity filter stays,
        so a shout over a cheering crowd does not become a clip and the shout class is often empty.
    4.  Tests the best candidates (up to 8) for Zero Crossing Rate (ZCR <= 0.15, after denoising) and keeps the first
        `REF_AUDITION_CANDIDATES` (3) that pass, best score first. When none passes, the tested candidate with the
        lowest ZCR is used: the speaker's own voice beats another speaker's reference. Which of the kept clips the TTS
        clones best is decided by the voice audition in Stage 5 (Spec 002, 2.3).
*   **Output:**
    *   `refs/{speaker_id}.wav`, `refs/{speaker_id}_2.wav`, ... (reference candidates for TTS, best score first).

### 2.5 Stage 5: Production (Global Task Pool)
*   **Action:** All segments from all `target_langs` are combined into a single task pool to maximize resource utilization.
*   **Workflow:**
    1.  **Draft Translation:** LLM generates drafts for all languages (see 2.5a).
    1b. **Designed voice and voice audition** (`TAKE_CHECK` `judge` or `whisper`): each speaker's reference candidates
        (Stage 4 clips, voice-bank clips, the designed voice) are tried on one of the speaker's own draft lines and
        ranked by the take judge's score (Spec 002, 2.3).
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
    translation reads as one text and terms, forms of address and gender stay consistent. Lines the answer leaves out,
    or "translates" by repeating an English line of three or more words, are asked again once, in two halves; only
    what is still missing keeps the source text (BattleBots S01E01: Gemma answered a whole window in English).
3.  **Information is moved, not dropped.** The translator must keep every piece of information of the window: names,
    numbers, results, reasons, who does what. Fillers ("you know", "well", repetitions) go first. When a line cannot
    hold everything within its `max_syllables`, the information may move to a neighbouring line of the same speaker that
    has spare budget. It must never move to another speaker.
4.  **Completeness review** (`TRANSLATION_REVIEW=on`, default). A second LLM pass compares each window's source and
    translation, lists the information that is missing or wrong and returns corrected lines within their budgets. Only
    the corrected lines are replaced; a correction far over its budget (more than 125% + 2 syllables) is ignored,
    except for a line the drafts left in the source language, which takes any translation (the sync loop shortens it;
    MythBusters S2012E19 kept "Now luckily, we've got just the guy to do that." because its fix was too long). A
    "correction" that repeats the source line is ignored.
5.  **Length variants.** Besides the main translation (aimed at `target_syllables`), the translator returns a `short`
    variant (towards `min_syllables`, essentials only) and a `long` one (towards `max_syllables`, fuller phrasing, no
    invented facts). All carry the same information as far as their length allows. The sync loop picks between them
    before synthesis (Spec 002, 2.2). A line fixed by the review keeps only the fixed text.
6.  Lines that are pure sound markers (`[music]`, `(laughter)`) are not dubbed.
7.  **Subtitles as a hint.** With target-language subtitles (2.3), each line to translate carries its `subtitle`
    (the cues aligned to it, empty when none). The translator follows the subtitle's names, terms, forms of address
    and gender forms and may reuse its wording, but keeps all information of the source (subtitles drop some) and the
    syllable budget (subtitles are written for reading, not for this timing). The review sees the subtitle too.
8.  **Terms.** Established translations of names, titles, ranks, ship and place names and technical terms
    (`Starfleet` -> `Gwiezdna Flota`) come from the series memory and, with target-language subtitles, from the
    episode itself (Spec 006, 5). The translator, the review and the duration refinement must use them. A term that is
    translated is no longer pronounced in English, so it leaves the phonetic glossary of that episode.

### 2.6 Stage 6: Final Mix
*   **Input:** `final_{lang}_{index}.wav` segments, `orig.wav`.
*   **Action:**
    1.  Merges all speech segments into a single track.
    2.  Mixes it over the **original** audio (not the Demucs accompaniment, which also removes part of the sound effects).
    3.  Sidechain ducking lowers the original while the dub speaks; a low-passed copy of the original ("ambient ghost")
        keeps the room sound.
*   **Output:**
    *   `final_{lang}.wav` (Mixed audio track, lossless; it is encoded once, when muxing).

### 2.7 Stage 7: Muxing
*   **Input:** Original video, `final_{lang}.wav` for each target language.
*   **Action:**
    1.  FFmpeg adds the dubbed tracks **in front of** the original audio, encoded once with `AUDIO_CODEC` (default
        AAC, which plays in browsers and on phones; `ac3`/`eac3` for AV receivers, or `opus`) and tagged with the
        ISO 639-2 language code (`pol`; MP4 drops two-letter codes) and the title `AI - <Language>` (in MP4 also the
        track's handler name, which players show instead). Every original stream is copied untouched: video, audio,
        subtitles and attachments (MKV fonts).
    2.  The result replaces the source file in place: the original is moved to `<file>.bak`, then the new file is moved
        to the original path (the backup is restored if that fails).
    3.  The series memory is updated (Spec 006, 2).
*   **Output:** the source file with the new tracks; `<file>.bak`.

### 2.8 Stage 8: Reporting & Cleanup
*   **Action:**
    1.  Calculates total processing time, stage durations and LLM throughput (tokens per second), and logs a
        performance report.
    2.  Deletes the task's folder (unless `DEBUG=1`). A failed task keeps it, so a retry reuses the checkpointed
        stages.
    3.  The worker sets the task status to `DONE`.

## 3. Data Persistence
-   **Task Table:** Tracks overall progress (`QUEUED`, `PROCESSING`, `DONE`, `ERROR`).
-   **Job Steps Table:** Stores the result of each stage (`result_data` JSON) to enable skipping completed steps on retry.
