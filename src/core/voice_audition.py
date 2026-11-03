"""Voice audition (Spec 002, 2.3): which of a speaker's reference clips OmniVoice clones intelligibly.

Each candidate synthesizes the same line of the speaker; Whisper scores how much the take sounds like its text.
A few syntheses per speaker before Stage 5 replace most of the retries the per-take check would need later.
"""

import logging
import os
from typing import Callable, Dict, List, Optional

from infrastructure.ffmpeg import FFmpegWrapper
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
