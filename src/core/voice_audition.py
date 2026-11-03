"""Voice audition (Spec 002, 2.3): which of a speaker's reference clips OmniVoice clones intelligibly.

Each candidate synthesizes the same line of the speaker; Whisper scores how much the take sounds like its text.
A few syntheses per speaker before Stage 5 replace most of the retries the per-take check would need later.
"""

import logging
import os
from typing import Callable, Dict, List, Optional

from infrastructure.ffmpeg import FFmpegWrapper
from infrastructure.tts_client import write_ref_text
from utils import count_syllables, text_similarity

logger = logging.getLogger(__name__)

# Test line length: long enough to hear the voice, short enough to synthesize quickly (about 4 s)
AUDITION_SYLLABLES = 18
MIN_AUDITION_SYLLABLES = 6


def pick_test_line(drafts: List[Dict], speaker: str, lang: str) -> Optional[str]:
    """The speaker's draft line closest to AUDITION_SYLLABLES syllables; None when the speaker has nothing to say."""
    texts = [(d.get("text") or "").strip() for d in drafts if d.get("speaker") == speaker]
    texts = [t for t in texts if count_syllables(t, lang) >= MIN_AUDITION_SYLLABLES]
    if not texts:
        return None
    return min(texts, key=lambda t: abs(count_syllables(t, lang) - AUDITION_SYLLABLES))


# OmniVoice voice design understands only these attributes (gender, age, pitch); Spec 002, 2.3
AGE_WORDS = [
    ("child", ("child", "kid", "boy ", "girl ")),
    ("teenager", ("teen",)),
    ("middle-aged", ("middle-aged", "middle aged", "40s", "50s")),
    ("elderly", ("elderly", "old man", "old woman", "senior")),
    ("young adult", ("young", "20s", "30s")),
]
PITCH_WORDS = [("low pitch", ("deep", "low-pitched", "low voice", "baritone", "bass")), ("high pitch", ("high-pitched",))]
# Designed clips are this long: within the 3-10 s OmniVoice clones best from
DESIGN_SYLLABLES = 30


def voice_instruct(info: Dict) -> Optional[str]:
    """OmniVoice voice-design attributes for a speaker profile ("male, middle-aged, low pitch"); None without a gender."""
    gender = (info.get("gender") or "").strip().lower()
    if gender not in ("male", "female"):
        return None
    desc = f" {(info.get('age') or '')} {(info.get('desc') or '')} ".lower()
    parts = [gender]
    for word, keys in AGE_WORDS:
        if any(k in desc for k in keys):
            parts.append(word)
            break
    for word, keys in PITCH_WORDS:
        if any(k in desc for k in keys):
            parts.append(word)
            break
    return ", ".join(parts)


def design_text(drafts: List[Dict], speaker: str, lang: str, avoid: Optional[str]) -> Optional[str]:
    """The speaker's own lines (not the audition test line) joined to about DESIGN_SYLLABLES syllables."""
    picked, total = [], 0
    for d in drafts:
        text = (d.get("text") or "").strip()
        if d.get("speaker") != speaker or not text or text == avoid:
            continue
        picked.append(text)
        total += count_syllables(text, lang)
        if total >= DESIGN_SYLLABLES:
            break
    return " ".join(picked) if total >= MIN_AUDITION_SYLLABLES else None


def design_voices(tts_manager, speaker_info: Dict[str, Dict], drafts: List[Dict], lang: str, work_dir: str) -> Dict[str, str]:
    """Synthesizes one voice-design clip per speaker from its profile and adds it as a reference candidate.

    Cloning from that clip keeps the voice the same across the episode (designing every line anew would not). The
    audition then decides whether it beats the speaker's own clips. Returns speaker -> clip path.
    """
    designed = {}
    tts_manager.load_engine()
    engine = tts_manager.engine
    if not engine or not hasattr(engine, "design"):
        return designed
    for spk, info in sorted((speaker_info or {}).items()):
        instruct = voice_instruct(info or {})
        text = design_text(drafts, spk, lang, pick_test_line(drafts, spk, lang)) if instruct else None
        if not text:
            continue
        out = os.path.join(work_dir, f"designed_{lang}_{spk}.wav")
        if tts_manager.service:
            ok = tts_manager.service.submit(engine.design, text, instruct, out, lang, priority=1).result()
        else:
            ok = engine.design(text, instruct, out, lang)
        if ok and os.path.exists(out) and os.path.getsize(out) > 4000:
            write_ref_text(out, text)
            tts_manager.add_designed_ref(spk, out)
            designed[spk] = out
            logger.info(f"Voice design ({lang}) {spk}: '{instruct}'")
        else:
            logger.warning(f"Voice design ({lang}) {spk}: no usable clip for '{instruct}'")
    return designed


def audition_voices(
    tts_manager,
    checker: Callable[[str, str], Optional[str]],
    drafts: List[Dict],
    lang: str,
    work_dir: str,
    is_swallowed: Callable[[str, float, str], bool],
    abort_event=None,
) -> Dict[str, Dict[str, float]]:
    """Scores every reference candidate of every speaker: speaker -> {reference path: similarity 0..1}.

    A speaker with a single candidate is auditioned too: a low score turns its line clips off and lets a same-gender
    voice go first (Spec 002, 2.3). Speakers without a usable test line, or whose takes cannot be checked, are left out.
    """
    scores: Dict[str, Dict[str, float]] = {}
    tts_manager.load_engine()
    for spk, refs in sorted(tts_manager.speaker_refs.items()):
        if abort_event is not None and abort_event.is_set():
            break
        text = pick_test_line(drafts, spk, lang)
        if not text or not refs:
            continue
        by_path: Dict[str, float] = {}
        for n, ref in enumerate(refs):
            if not os.path.exists(ref):
                continue
            out = os.path.join(work_dir, f"audition_{lang}_{spk}_{n}.wav")
            try:
                tts_manager._run_synthesis(text, ref, out, language=lang)
                duration = FFmpegWrapper.get_duration(out) if os.path.exists(out) else 0.0
                if duration <= 0 or is_swallowed(text, duration, lang):
                    score, heard = 0.0, ""
                else:
                    heard = checker(out, lang)
                    if not isinstance(heard, str):
                        continue
                    score = text_similarity(text, heard)
            except Exception as e:
                logger.warning(f"Voice audition: {spk} candidate {n + 1} failed: {e}")
                score, heard = 0.0, ""
            finally:
                if os.path.exists(out):
                    os.remove(out)
            by_path[ref] = round(score, 2)
            logger.debug(f"Voice audition: {spk} candidate {n + 1} heard '{heard[:60]}'")
        if by_path:
            scores[spk] = by_path
            ranked = ", ".join(f"{os.path.basename(p)} {s:.2f}" for p, s in by_path.items())
            logger.info(f"Voice audition ({lang}) {spk}: {ranked} | test line '{text[:50]}'")
    return scores
