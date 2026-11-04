"""Subtitle files: finding sidecars by language, reading SRT cues and aligning them to script lines (Spec 001, 2.3)."""

import glob
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ISO_639_2 = {
    "pl": "pol",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "cs": "ces",
    "uk": "ukr",
    "nl": "nld",
    "sv": "swe",
    "ja": "jpn",
    "zh": "zho",
    "en": "eng",
}
# Bibliographic 639-2 codes that differ from the terminology ones above
ISO_639_2B = {"deu": "ger", "fra": "fre", "ces": "cze", "nld": "dut", "zho": "chi"}

SDH_TAGS = ("hi", "sdh", "cc")
MIN_CUE_OVERLAP = 1 / 3  # share of a cue's duration that must lie inside a line

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_TAGS = re.compile(r"<[^>]+>|\{\\[^}]*\}")
# Credits of the people who made the subtitles: never spoken (Spec 001, 2.3). A web address or domain, or the words
# subtitle makers sign with ("Sync and corrections by", "Tłumaczenie:", "Original air date").
_CREDIT = re.compile(
    r"(https?://|www\.|\b[\w-]+\.(com|org|net|info|tv|pl|de|ru|eu|io|cc|me)\b"
    r"|\b(subtitles?|subs|captions?|captioning|sync(ed|hro)?|corrections?|corrected|transcript|ripped|encoded"
    r"|translated|translation|resync|timing)\s+(by|and)\b"
    r"|\b(addic7ed|opensubtitles|subscene|podnapisi|napiprojekt|napisy24|titlovi|yify|tvsubtitles)\b"
    r"|\boriginal air ?date\b|\b(tłumaczenie|przekład|synchro(nizacja)?|korekta|napisy)\s*:)",
    re.IGNORECASE,
)


def is_credit(text: str) -> bool:
    """A cue that credits the subtitle makers or advertises a site: "Sync and corrections by X for www.addic7ed.com"."""
    return bool(_CREDIT.search(text or ""))


def lang_codes(lang: str) -> set:
    """Every file-name code of a language: `pl`, `pol`, and the bibliographic code where it differs."""
    lang = (lang or "").lower()
    codes = {lang}
    for two, three in ISO_639_2.items():
        if lang in (two, three, ISO_639_2B.get(three)):
            codes |= {two, three, ISO_639_2B.get(three, three)}
    return {c for c in codes if c}


def find_sidecar(
    vpath: str, langs: Optional[Iterable[str]] = None, exclude: Iterable[str] = (), sdh: bool = False
) -> Optional[str]:
    """Best sidecar `<video>[.<lang>][.hi].srt`: in one of ``langs`` (None: any language not in ``exclude``).

    Forced (partial) subtitles are skipped; full subtitles beat SDH ones (``sdh``: only SDH ones). Without ``langs`` an
    untagged `<video>.srt` comes first, then English.
    """
    base = os.path.splitext(vpath)[0]
    wanted = set().union(*(lang_codes(x) for x in langs)) if langs else None
    excluded = set().union(*(lang_codes(x) for x in exclude)) if exclude else set()
    candidates = []
    for path in glob.glob(glob.escape(base) + "*.srt"):
        if path[len(base) : len(base) + 1] not in ("", "."):
            continue  # another video whose name starts like this one
        tags = [t.lower() for t in path[len(base) : -len(".srt")].split(".") if t]
        if "forced" in tags or any(t in excluded for t in tags):
            continue
        is_sdh = 1 if any(t in SDH_TAGS for t in tags) else 0
        if sdh and not is_sdh:
            continue
        if wanted is not None:
            if not any(t in wanted for t in tags):
                continue
            rank = (is_sdh, 0)
        else:
            rank = (0 if not tags else 1 if tags[0] in ("en", "eng") else 2, is_sdh)
        candidates.append((rank, path))
    return min(candidates)[1] if candidates else None


def read_text(path: str) -> str:
    """A subtitle file's text: UTF-8 (with or without BOM), else Windows-1250, the usual encoding of Polish subtitles."""
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp1250"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _seconds(match) -> float:
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> List[Tuple[float, float, str]]:
    """[(start, end, text)] of an SRT; formatting tags dropped, lines of one cue joined."""
    cues = []
    for block in re.split(r"\n\s*\n", (text or "").replace("\r\n", "\n").replace("\r", "\n")):
        lines = [x.strip() for x in block.strip().split("\n") if x.strip()]
        for i, line in enumerate(lines):
            if "-->" not in line:
                continue
            times = list(_TIME.finditer(line))
            if len(times) < 2:
                break
            body = " ".join(_TAGS.sub("", x).strip() for x in lines[i + 1 :])
            body = re.sub(r"\s+", " ", body).strip()
            if body and is_credit(body):
                logger.info(f"Subtitles: credit cue left out: '{body[:80]}'")
            elif body:
                cues.append((_seconds(times[0]), _seconds(times[1]), body))
            break
    return cues


def best_line(script: List[Dict], start: float, end: float) -> Optional[int]:
    """Position in ``script`` of the line a cue overlaps most, when that is at least MIN_CUE_OVERLAP of the cue."""
    length = max(end - start, 0.01)
    best, best_overlap = None, 0.0
    for i, line in enumerate(script):
        overlap = min(end, line["end"]) - max(start, line["start"])
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    return best if best is not None and best_overlap >= MIN_CUE_OVERLAP * length else None


def align(script: List[Dict], cues: List[Tuple[float, float, str]]) -> Dict[int, str]:
    """{script index: the cues spoken during that line}. Each cue goes to the line it overlaps most in time (a cue is
    never repeated on two lines)."""
    out: Dict[int, List[str]] = {}
    for start, end, text in cues:
        pos = best_line(script, start, end)
        if pos is not None:
            out.setdefault(script[pos].get("index", pos), []).append(text)
    return {i: " ".join(texts) for i, texts in out.items()}


# --- What else the subtitles tell (Spec 001, 2.3) ---

# SDH speaker labels: "KIRK:", "- SPOCK:", "[MCCOY]" at the start of a cue or of a dialogue line in it
_SDH_LABEL = re.compile(r"^\s*-?\s*(?:\[([A-Z][A-Z .'\-]{1,30})\]|([A-Z][A-Z .'\-]{1,30}):)\s*")
_SOUND = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪?|[♪#]")


def strip_sdh(text: str) -> Tuple[List[str], str]:
    """(speaker labels, the spoken text) of a cue: labels and sound descriptions ("[door opens]", "♪") removed."""
    names = []
    parts = []
    for part in re.split(r"(?:^|\s)(?=-\s)", text):
        m = _SDH_LABEL.match(part)
        if m:
            names.append((m.group(1) or m.group(2)).strip().title())
            part = part[m.end() :]
        parts.append(part)
    spoken = _SOUND.sub(" ", " ".join(parts))
    spoken = re.sub(r"(^|\s)-\s+", r"\1", spoken)
    return names, re.sub(r"\s+", " ", spoken).strip()


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def timing_matches(segments: List[Dict], cues: List[Tuple[float, float, str]], min_share: float = 0.7) -> bool:
    """Whether the subtitles are timed to this audio: most spoken cues overlap the transcription (another release of
    the video is often shifted by seconds, and its cues would then fill the wrong gaps)."""
    spoken = [(s, e) for s, e, t in cues if strip_sdh(t)[1]]
    if len(spoken) < 10 or not segments:
        return False
    hit = sum(1 for s, e in spoken if sum(_overlap(s, e, g["start"], g["end"]) for g in segments) >= MIN_CUE_OVERLAP * (e - s))
    return hit >= min_share * len(spoken)


def missing_speech(segments: List[Dict], cues: List[Tuple[float, float, str]], min_seconds: float = 1.0) -> List[Dict]:
    """Subtitle cues that Whisper did not hear: spoken cues of at least ``min_seconds`` that no transcribed segment
    overlaps by MIN_CUE_OVERLAP of the cue, as transcription segments {start, end, text, from_subtitles}."""
    out = []
    for start, end, text in cues:
        _, spoken = strip_sdh(text)
        length = end - start
        if not spoken or length < min_seconds:
            continue
        heard = sum(_overlap(start, end, g["start"], g["end"]) for g in segments)
        if heard < MIN_CUE_OVERLAP * length:
            out.append({"start": start, "end": end, "text": spoken, "avg_logprob": -0.5, "from_subtitles": True})
    return out


def speaker_labels(script: List[Dict], cues: List[Tuple[float, float, str]]) -> Dict[str, Dict[str, int]]:
    """{speaker: {SDH label: cues}}, counting the cues with a single label by the line they overlap most."""
    out: Dict[str, Dict[str, int]] = {}
    for start, end, text in cues:
        names, _ = strip_sdh(text)
        if len(set(names)) != 1:
            continue
        pos = best_line(script, start, end)
        if pos is None:
            continue
        counts = out.setdefault(script[pos]["speaker"], {})
        counts[names[0]] = counts.get(names[0], 0) + 1
    return out


def label_hints(labels: Dict[str, Dict[str, int]], min_cues: int = 2, min_share: float = 0.6) -> str:
    """The SDH speaker names for the analysis prompt: a name carried by most labelled cues of a speaker."""
    lines = []
    for spk, counts in sorted(labels.items()):
        total = sum(counts.values())
        name, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= min_cues and n >= min_share * total:
            lines.append(f"- {spk} is labelled {name} in {n} of {total} labelled subtitle lines")
    return "Speaker labels of the hearing-impaired subtitles:\n" + "\n".join(lines) if lines else ""


# Gendered first-person forms of Polish: past tense and conditional ("zrobiłem"/"zrobiłam", "zrobiłbym"/"zrobiłabym").
# "-łem" is also the instrumental of nouns in "-ł" ("kołem", "admirałem"): no "-ołem", and the common nouns are listed.
_PL_MALE = re.compile(r"\b\w+(?:[iyąęaeu]łem|[dgsz]łem|łbym)\b", re.IGNORECASE)
_PL_FEMALE = re.compile(r"\b\w{2,}(?:łam|łabym)\b", re.IGNORECASE)
_PL_NOUNS = {
    "admirałem",
    "generałem",
    "kapitałem",
    "kanałem",
    "materiałem",
    "działem",
    "upałem",
    "zapałem",
    "szałem",
    "kawałem",
    "ideałem",
    "kryształem",
    "sygnałem",
    "potencjałem",
    "finałem",
    "rytuałem",
    "arsenałem",
    "personelem",
    "mułem",
    "hasłem",
    "masłem",
    "krzesłem",
    "pomysłem",
    "zmysłem",
    "przemysłem",
    "namysłem",
    "posłem",
    "wiosłem",
    "rzemiosłem",
    "węzłem",
    "kozłem",
    "orłem",
    "osłem",
    "odłam",
    "przełam",
    "połam",
    "wyłam",
}
GRAMMAR_LANGS = ("pl",)


def grammar_gender(texts: Iterable[str], lang: str, min_forms: int = 3, min_share: float = 0.8) -> Optional[str]:
    """The speaker's gender from first-person forms in its target-language subtitles, or None when they do not tell.

    A dialogue cue ("- ... - ...") holds two speakers and is skipped.
    """
    if lang not in GRAMMAR_LANGS:
        return None
    male = female = 0
    for text in texts:
        if not text or re.search(r"\S\s+-\s", text.strip()):  # "- Byłem. - A ja byłam.": two speakers
            continue
        male += sum(1 for w in _PL_MALE.findall(text) if w.lower() not in _PL_NOUNS)
        female += sum(1 for w in _PL_FEMALE.findall(text) if w.lower() not in _PL_NOUNS)
    total = male + female
    if total < min_forms:
        return None
    if male >= min_share * total:
        return "male"
    if female >= min_share * total:
        return "female"
    return None
