"""Script verification and energy classes (Spec 001, 2.2c and 2.4; Spec 002, 2.3).

What Whisper is unsure of is heard again, every segment's speaker is checked by its voiceprint, every line gets an
energy class (calm / lively / shout) relative to its speaker, and a shouting cluster is merged with the same person's
calm one. Everything here works on 16 kHz mono samples of the vocals, read once.
"""

import logging
import math
import os
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from config import (
    ASR_UNSURE_LOGPROB,
    ASR_UNSURE_WORD,
    SPEAKER_RECHECK_MARGIN,
    SPEAKER_RECHECK_MIN,
    SHOUT_MERGE_MIN,
)
from utils import count_syllables

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy comes with the app
    np = None

logger = logging.getLogger(__name__)

RATE = 16000
# A segment this long or longer gets a voiceprint for the speaker check
RECHECK_MIN_SECONDS = 1.5
# Context cut around an unsure segment for the second hearing
HEAR_PAD = 0.3
# Energy classes, ordered by arousal
ENERGY_CLASSES = ("calm", "lively", "shout")
SHOUT_SCORE = 6.0
LIVELY_SCORE = 2.5
FAST_PACE = 1.3
# One person split in two clusters: the louder one this much louder and higher (Spec 002, 2.3)
SHOUT_MERGE_DB = 3.0
SHOUT_MERGE_SEMITONES = 3.0


# --- audio ---


def load_samples(path: str, work_dir: str) -> Optional["np.ndarray"]:
    """The file as 16 kHz mono float samples; None when it cannot be read."""
    from core.take_judge import read_wav_mono

    if np is None or not path or not os.path.exists(path):
        return None
    out = os.path.join(work_dir, "check_16k.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(RATE), out, "-y"],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"Script check: could not read {path}: {e}")
        return None
    data = read_wav_mono(out)
    try:
        os.remove(out)
    except OSError:
        pass
    return data[0] if data else None


def span(samples, start: float, end: float):
    if samples is None:
        return None
    lo, hi = max(0, int(start * RATE)), min(len(samples), int(end * RATE))
    return samples[lo:hi] if hi > lo else None


def level_db(samples) -> Optional[float]:
    """Level (dB) of the louder half of the frames: the voice, not the pauses between words."""
    if samples is None or np is None or len(samples) < RATE // 10:
        return None
    frame = RATE // 50
    n = len(samples) // frame
    if n == 0:
        return None
    energy = np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1)
    loud = np.sort(energy)[n // 2 :]
    mean = float(np.mean(loud))
    return 10 * math.log10(mean) if mean > 0 else None


def semitones(hz: float, ref: float) -> float:
    return 12 * math.log2(hz / ref)


# --- 1. confidence ---


def is_unsure(segment: Dict) -> bool:
    """Whisper's confidence in a segment is low (Spec 001, 2.2c step 1)."""
    if segment.get("avg_logprob", 0.0) < ASR_UNSURE_LOGPROB:
        return True
    word = segment.get("min_word_prob")
    return word is not None and word < ASR_UNSURE_WORD


# --- 2. second hearing ---


def subtitle_text(cues: List[Tuple[float, float, str]], start: float, end: float) -> str:
    """The reference cues spoken during a span (at least a third of each cue inside it)."""
    out = []
    for c_start, c_end, text in cues or []:
        overlap = min(end, c_end) - max(start, c_start)
        if overlap > 0 and overlap >= (c_end - c_start) / 3:
            out.append(text.strip())
    return " ".join(t for t in out if t)


def second_hearing(
    segments: List[Dict],
    transcribe: Callable[["np.ndarray", Optional[str]], str],
    original,
    vocals,
    names: List[str],
    cues=None,
) -> List[Dict]:
    """Hears the unsure segments again (Spec 001, 2.2c step 2) and returns the segments with their ``hearings``.

    ``transcribe(samples, prompt)`` is Whisper on 16 kHz samples; ``original`` and ``vocals`` are such samples.
    """
    out = []
    names_text = ", ".join(n for n in names if n)[:400]
    unsure = 0
    for i, seg in enumerate(segments):
        seg = dict(seg)
        if is_unsure(seg):
            unsure += 1
            hearings = {}
            start, end = max(0.0, seg["start"] - HEAR_PAD), seg["end"] + HEAR_PAD
            clip = span(vocals, start, end)
            if clip is not None:
                hearings["vocals"] = transcribe(clip, None)
            clip = span(original, start, end)
            if clip is not None:
                previous = segments[i - 1]["text"] if i > 0 else ""
                prompt = " ".join(p for p in (names_text and f"Names: {names_text}.", previous) if p)
                hearings["prompted"] = transcribe(clip, prompt or None)
            subtitle = subtitle_text(cues, seg["start"], seg["end"])
            if subtitle:
                hearings["subtitle"] = subtitle
            hearings = {k: v.strip() for k, v in hearings.items() if v and v.strip() and v.strip() != seg["text"].strip()}
            seg["unsure"] = True
            if hearings:
                seg["hearings"] = hearings
        out.append(seg)
    logger.info(f"Script check: {unsure} of {len(segments)} segments unsure, heard again")
    return out


def line_hearings(parts: List[Dict]) -> Dict[str, str]:
    """A line's other hearings, from its segments: each kind with the segments that have none kept as they are."""
    kinds = sorted({k for p in parts for k in (p.get("hearings") or {})})
    out = {}
    for kind in kinds:
        out[kind] = " ".join(((p.get("hearings") or {}).get(kind) or p["text"]).strip() for p in parts).strip()
    return out


# --- 3. who speaks ---


def recheck_speakers(
    segments: List[Dict],
    centroids: Dict[str, List[float]],
    voiceprint: Callable[["np.ndarray"], Optional["np.ndarray"]],
    vocals,
) -> List[Dict]:
    """Moves a segment to the speaker its voice is clearly closer to (Spec 001, 2.2c step 3).

    Each segment carries the speaker it overlaps most (``speaker``); the result sets ``speaker`` and, when moved,
    ``speaker_moved_from``.
    """
    if np is None or not centroids or vocals is None:
        return segments
    names = [s for s, c in centroids.items() if c]
    mat = np.asarray([centroids[s] for s in names], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms > 0, norms, 1)
    out, moved = [], 0
    for seg in segments:
        seg = dict(seg)
        spk = seg.get("speaker")
        if spk in names and seg["end"] - seg["start"] >= RECHECK_MIN_SECONDS:
            emb = voiceprint(span(vocals, seg["start"], seg["end"]))
            if emb is not None:
                sims = mat @ np.asarray(emb, dtype=np.float32)
                best = int(np.argmax(sims))
                own = float(sims[names.index(spk)])
                if names[best] != spk and sims[best] >= SPEAKER_RECHECK_MIN and sims[best] - own >= SPEAKER_RECHECK_MARGIN:
                    logger.info(
                        f"Speaker check: [{seg['start']:.1f}-{seg['end']:.1f}] {spk} -> {names[best]} "
                        f"({own:.2f} vs {float(sims[best]):.2f}) '{seg.get('text', '')[:50]}'"
                    )
                    seg["speaker_moved_from"] = spk
                    seg["speaker"] = names[best]
                    moved += 1
        out.append(seg)
    logger.info(f"Speaker check: {moved} segments moved to another speaker")
    return out


# --- energy classes ---


def measure_lines(script: List[Dict], vocals, pitch: Callable) -> Dict[int, Dict]:
    """Level (dB), pitch (Hz) and pace (syllables/s) of each line on the vocals, by line index."""
    out = {}
    for line in script:
        clip = span(vocals, line["start"], line["end"])
        dur = line["end"] - line["start"]
        out[line["index"]] = {
            "level": level_db(clip),
            "pitch": pitch(clip, RATE) if clip is not None else None,
            "pace": count_syllables(line.get("text_en", ""), "en") / dur if dur > 0 else None,
        }
    return out


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def energy_class(m: Dict, med: Dict) -> Optional[str]:
    """calm / lively / shout of a line against its speaker's medians (Spec 001, 2.4 step 3)."""
    if m.get("level") is None or med.get("level") is None:
        return None
    score = m["level"] - med["level"]
    if m.get("pitch") and med.get("pitch"):
        score += semitones(m["pitch"], med["pitch"])
    if m.get("pace") and med.get("pace") and m["pace"] >= FAST_PACE * med["pace"]:
        score += 1
    if score >= SHOUT_SCORE:
        return "shout"
    if score >= LIVELY_SCORE:
        return "lively"
    return "calm"


def speaker_medians(script: List[Dict], measures: Dict[int, Dict]) -> Dict[str, Dict]:
    by_spk: Dict[str, List[Dict]] = {}
    for line in script:
        if line["index"] in measures:
            by_spk.setdefault(line["speaker"], []).append(measures[line["index"]])
    return {spk: {k: _median(m[k] for m in ms) for k in ("level", "pitch", "pace")} for spk, ms in by_spk.items()}


def assign_energy(script: List[Dict], measures: Dict[int, Dict]) -> Dict[str, Dict[str, int]]:
    """Sets each line's ``energy``; returns speaker -> {class: lines} for the log."""
    medians = speaker_medians(script, measures)
    counts: Dict[str, Dict[str, int]] = {}
    for line in script:
        m = measures.get(line["index"])
        cls = energy_class(m, medians.get(line["speaker"], {})) if m else None
        line["energy"] = cls
        if cls:
            counts.setdefault(line["speaker"], {}).setdefault(cls, 0)
            counts[line["speaker"]][cls] += 1
    return counts


def class_distance(a: Optional[str], b: Optional[str]) -> int:
    """Steps between two energy classes; an unknown class is one step from anything."""
    if a not in ENERGY_CLASSES or b not in ENERGY_CLASSES:
        return 1
    return abs(ENERGY_CLASSES.index(a) - ENERGY_CLASSES.index(b))


# --- one person, two clusters ---


def _cosine(a, b) -> float:
    if np is None or not a or not b:
        return 0.0
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na and nb else 0.0


def shout_merges(
    script: List[Dict],
    measures: Dict[int, Dict],
    speaker_info: Dict[str, Dict],
    prints: Dict[str, List[float]],
    measured_genders: Dict[str, str],
) -> Dict[str, str]:
    """{shouting speaker: calm speaker} for clusters of one person split by how loud they speak (Spec 002, 2.3)."""
    medians = speaker_medians(script, measures)

    def name(spk):
        n = str((speaker_info.get(spk) or {}).get("name") or "").strip().lower()
        return "" if n in ("", "unknown", "none", "null") else n

    merges: Dict[str, str] = {}
    speakers = sorted(s for s in medians if name(s))
    for loud in speakers:
        for calm in speakers:
            if loud == calm or loud in merges or calm in merges or name(loud) != name(calm):
                continue
            ml, mc = medians[loud], medians[calm]
            if None in (ml["level"], mc["level"], ml["pitch"], mc["pitch"]):
                continue
            if ml["level"] - mc["level"] < SHOUT_MERGE_DB or semitones(ml["pitch"], mc["pitch"]) < SHOUT_MERGE_SEMITONES:
                continue
            g_loud, g_calm = measured_genders.get(loud), measured_genders.get(calm)
            if g_loud and g_calm and g_loud != g_calm:
                continue
            sim = _cosine(prints.get(loud), prints.get(calm))
            if sim < SHOUT_MERGE_MIN:
                logger.info(f"Shout merge: {loud} and {calm} are both '{name(loud)}', but voices only {sim:.2f} alike")
                continue
            logger.info(
                f"Shout merge: {loud} ({ml['level'] - mc['level']:+.1f} dB, "
                f"{semitones(ml['pitch'], mc['pitch']):+.1f} st, voice {sim:.2f}) is {calm} shouting: merged"
            )
            merges[loud] = calm
    return merges
