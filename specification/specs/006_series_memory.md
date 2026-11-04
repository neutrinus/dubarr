# Spec 006: Series Memory

## 1. Goal
Episodes of one series are dubbed one after another. Each episode should sound **consistent with the previous ones**
(the same people with the same voices, the same names pronounced the same way) and **better** for what was learned
from them. Everything happens automatically: the user never curates voices or names.

Implemented in `src/core/series_memory.py`; used by Stage 3 (glossary), Stage 4 and the voice audition (voices).

## 2. Storage
-   `SERIES_MEMORY` (`on` by default; `off` disables reading and writing).
-   Folder `DATA_DIR/series/<key>/`, where `<key>` is the series title from the media metadata (Spec 001, 2.2b),
    lower-cased, with everything but letters and digits replaced by `-` (`battlebots`). Movies have no series memory.
-   `glossary.json`, `terms.json`, `voices.json` and `clips/`. Files are written atomically (temp file + rename) at the end of a
    successful task; a failed task leaves the memory as it was.
-   Two tasks of the same series never run at the same time (the worker runs one task at a time).

## 3. Glossary
The analysis (Spec 001, 2.3) writes a **phonetic respelling** of names that keep their original sound, so the TTS
pronounces "Razorback" in English ("Rejzorbek"). Without memory it was invented again on every run and never checked
("Rejzor-bak" in one run, "Rej-or-bek", which drops the "z", in the next).

1.  `glossary.json` maps `{lang key: {term: {"spelling": str, "checked": bool, "episodes": int}}}`; the lang key is
    the comma-joined target languages of the analysis (`pl`), because one analysis writes one glossary for them.
2.  **Known names first.** Before the analysis, the names in the episode title (the Sonarr file name or TVmaze,
    split at `vs.`, `vs`, `,`, `&` and ` and `: "Backlash vs. Disposable Hero, Tentoumushi vs. Sallad") and the cast
    names join the prompt as names that occur in this episode, so the corrector of the transcript can match a misheard
    name against them.
3.  **Fixed entries.** Terms already in the memory are given to the analysis as fixed: the model must reuse them as they
    are and only add new terms. Its answer cannot change a stored spelling; only the check by spelling rules (5)
    can, when the stored spelling fails it.
4.  **Reading test** for every new or changed spelling that differs from the term: a separate short LLM request gets
    only the spellings (not the terms) and writes, in English spelling, what an English speaker would hear when a
    native speaker of the target language reads each of them aloud. The answer is compared with the term
    by their consonants: letters only, doubled letters once, voiced and voiceless pairs folded (`b`/`p`, `d`/`t`,
    `g`/`k`, `z`/`s`, `v`/`f`/`w`), `ck`/`ch`/`c`/`q` -> `k`, `ph` -> `f`, `th` -> `t`, and `j`/`y` counted as vowels.
    A spelling that loses or changes more than one consonant in six fails ("Rayorbeck" keeps `rrpk` of `rsrpk`): the
    model is asked for a new spelling of the failed terms, with the failed spelling and what it sounded like, at most
    twice. A term that still fails keeps its original spelling (the TTS reads it as written, which beats a wrong name).
5.  **Check by spelling rules** (Polish target): the LLM writes each name's English pronunciation in CMU ARPAbet
    once (stored under the key `cmu`, shared with 7), and the respelling is read by Polish spelling rules: `w` is
    V, `ł` is W, `sz`/`ś`/`si`+vowel SH, `cz`/`ć`/`ci`+vowel CH, `rz`/`ż`/`ź` ZH, `dż`/`dź`/`dzi` JH, `c` TS,
    `ch`/`h` HH, `j` and the vowels no consonant. The consonants of both are aligned (voiced and voiceless pairs
    folded, ER counts as R, TH and DH match T, F or S, a consonant the spelling adds costs little). A Polish `w`
    standing for an English W becomes `ł` in place, and an `ł` for an English V becomes `w` ("Ajs Wejv" ->
    "Ajs Łejv", "Worehed" -> "Łorehed"): the model wrote `w` for W in nearly every name of the first measured run,
    and its own reading test passed them. Then one lost or changed consonant per six may remain, W read as V
    never. A failed spelling is respelled as in 4, the respelling prompt carrying the Polish rules (English w is
    `ł`, v is `w`, sh `sz`, ch `cz`, j `dż`, y `j`, th `t` or `f`) and what was lost. Stored spellings go through
    this check too; one that changes or fails is replaced in the memory. Names without phonemes, and other target
    languages, keep the LLM reading test of 4.
6.  Acronyms of up to 6 capitals (`S.T.C.W.R.`) skip the test. Only checked entries are stored. A stored entry counts
    the episodes that used it.

7.  **Phonemes instead** (`GLOSSARY_MODE=phonemes`, an experiment): the translation keeps the original names, the
    LLM writes each name's English pronunciation in CMU ARPAbet (`R EY1 Z ER0 B AE2 K`; answers with anything but
    ARPAbet tokens are dropped), and the TTS text gets the phonemes in brackets in place of the name, a Polish ending
    staying after them (`[R EY1 Z ER0 B AE2 K]owi`). OmniVoice documents bracketed CMU phonemes for English text
    only, so this must be compared with the respellings by ear before it can become the default. The phonemes are
    stored in the series glossary under the key `cmu`.

## 4. Voice Bank
1.  **Voiceprint.** Diarization keeps each speaker's centroid embedding, which pyannote computes anyway (Spec 001,
    2.2). Reference clips and takes are embedded with the same model (`pyannote/wespeaker-voxceleb-resnet34-LM`), so
    all prints compare by cosine similarity.
2.  **`voices.json`** holds one entry per person: id (`V001`...), the mean voiceprint over its episodes, name, gender,
    a short description, the number of episodes, and up to `VOICE_BANK_CLIPS` (5) clips in `clips/` (with their
    transcripts), their audition score and their quality (0..1: half the length's fit to 6-10 s, half the clarity
    in dB when the clip was cut, Spec 001, 2.4) and its energy class (`calm`, `lively`, `shout`; Spec 001, 2.4).
    With `EMOTION_REFS` the slots are per class: up to `VOICE_BANK_CLASS_CLIPS` (2) clips of each class, so a
    person who only shouts in one episode keeps calm clips from another, and the reverse. Clips are ranked by audition score plus 0.1 times the quality, so
    between clips that clone about equally well the longer and cleaner one stays; the worst is dropped when a better
    one arrives. Every bank clip of a matched person is auditioned again in each episode, on that episode's test line
    next to the episode's own clips, so stored scores stay comparable with new ones and the bank only gets better. Only auditioned clips are stored (none with `TAKE_CHECK=off`), and never a designed voice.
    A person merged from a shouting and a calm cluster (Spec 002, 2.3) also keeps the shouting cluster's voiceprint
    (`shout_print`); matching takes the better of the two, so next episode the shouting cluster finds the person
    instead of a new or duplicate entry.
3.  **Matching.** After Stage 2 every diarization speaker is compared with the bank. A speaker is that person when the
    similarity is at least `VOICE_MATCH_MIN` (0.6) and beats the second best person by at least 0.05 (two similar
    commentators must not merge). Two speakers that match the same person are merged into one speaker before the script
    is built, so a person split by the diarization speaks with one voice. A speaker without a match is new.
4.  **Profiles.** The analysis gets the matched persons as hints ("SPEAKER_03 is probably <name>, male, commentator,
    known from 4 episodes"). A gender measured from the voice's pitch (Spec 002, 2.3) overrules the analysis and is
    stored with that pitch; a measured gender known from at least 2 episodes is not guessed again. A gender that was
    only guessed is not repeated in the hints and is replaced by the latest analysis, since S01E03 stored three men as
    women from their shouting pitch. A measured gender that contradicts the stored one also drops the stored name:
    the person was taken for someone else.
5.  **References.** A matched speaker's bank clips join its reference candidates (Spec 001, 2.4) before the voice
    audition, which ranks all of them as usual: the bank cannot make a speaker worse, since a better clip of this
    episode wins. A speaker who only shouts in this episode gets a calm clip from an earlier one.
6.  **Update.** After a successful task, every speaker with at least 3 lines updates its person (or creates a
    new one): the voiceprint is averaged (weighted by episodes), the profile is filled where it was empty, and the
    episode's audited clips compete for the bank's slots.

## 5. Terms
Names and terms that the target language translates rather than pronounces (`Starfleet` -> `Gwiezdna Flota`,
`phaser` -> `fazer`, ranks, titles, ship and place names). A human translation knows them; the LLM guesses them anew in
every episode.

1.  **From the episode's subtitles.** When the episode has target-language subtitles (Spec 001, 2.3), the LLM reads the
    script lines next to their aligned subtitles, in chunks of 60 lines, and lists the pairs that the subtitles
    translate: proper names, titles, ranks, ship, place and organisation names, technical and invented terms. Not
    common words, and not names the subtitles leave as they are (those belong to the glossary, 3). A pair is kept only
    when the target term occurs in the subtitles and the source term in the script.
2.  **`terms.json`** maps `{lang: {term: {"translation": str, "episodes": int}}}`, per target language. After a
    successful task the episode's terms are added; a term the subtitles of this episode translate differently replaces
    the stored one (the latest human translation wins), and every term seen again is counted.
3.  **Forms of address.** The same request lists how one named speaker addresses another in the subtitles
    (`Spock -> Kirk: formal, "kapitanie", "pan"`). They are stored next to the terms (`address`), the latest episode
    winning, and given to the translator with the terms.
4.  **Use.** Every episode gets the stored terms, with the terms of its own subtitles on top, for the translation
    (Spec 001, 2.5a), so an episode without subtitles uses the names of the episodes that had them.

## 6. Configuration
-   `SERIES_MEMORY` (on): series glossary and voice bank.
-   `GLOSSARY_MODE` (respell): `respell` (phonetic spelling with the reading test) or `phonemes` (CMU phonemes).
-   `VOICE_MATCH_MIN` (0.6): voiceprint similarity from which a speaker is a known person. To be tuned on measured runs.
-   `VOICE_BANK_CLIPS` (5): clips kept per person; `VOICE_BANK_CLASS_CLIPS` (2): per energy class with `EMOTION_REFS`.
