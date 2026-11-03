import logging
import os
import shutil
import wave
from typing import Dict, List, Optional
from infrastructure.ffmpeg import FFmpegWrapper
from config import MAX_ACCEPT_SPEEDUP, MIN_FILL_RATIO, MAX_PLAUSIBLE_SYLLABLES_PER_SECOND, TAKE_MIN_SIMILARITY
from utils import count_syllables, text_similarity
from duration_policy import MAX_SPEEDUP as TTS_MAX_SPEEDUP

logger = logging.getLogger(__name__)


# Extra syntheses of the same text after a broken take (swallowed or unintelligible), each with another voice reference
SWALLOW_RETRIES = 2


class SegmentSynchronizer:
    def __init__(self, llm_manager, tts_manager, temp_dir, debug_mode=False, take_checker=None):
        self.llm = llm_manager
        self.tts = tts_manager
        self.temp_dir = temp_dir
        self.debug_mode = debug_mode
        # (audio_path, lang) -> what an ASR hears in the take, or None when it cannot tell (Spec 002, 2.2 step 4)
        self.take_checker = take_checker

    @staticmethod
    def is_swallowed(text: str, duration: float, lang: str) -> bool:
        """True when a take is far shorter than its text can be spoken: the TTS produced mostly silence."""
        syllables = count_syllables(text, lang)
        return syllables >= 3 and duration * MAX_PLAUSIBLE_SYLLABLES_PER_SECOND < syllables

    def _intelligibility(self, text: str, audio_path: str, lang: str):
        """(similarity 0..1, heard text) of a take; None without a checker or when it cannot tell."""
        if not self.take_checker:
            return None
        try:
            heard = self.take_checker(audio_path, lang)
        except Exception as e:
            logger.warning(f"Take check failed: {e}")
            return None
        if not isinstance(heard, str):
            return None
        return text_similarity(text, heard), heard

    def _synthesize_sound(self, tts_item: Dict, lang: str, vocals_path: str, full_script: List[Dict]):
        """Synthesizes a take and re-synthesizes it with other references while it comes out broken.

        Broken means swallowed (far too short for its text) or unintelligible (the ASR hears something else: noise,
        screams, mangled words). Returns (result, duration, broken) or None when synthesis failed; when every reference
        gives a broken take, the least broken one is returned.
        """
        idx = tts_item["index"]
        text = tts_item["text"]
        skip = []
        best = None  # (score, (result, duration, True)); higher score = less broken
        for retry in range(SWALLOW_RETRIES + 1):
            item = dict(tts_item, skip_refs=list(skip))
            result = self.tts.synthesize_sync(item, lang, vocals_path, full_script)
            if not result:
                return best[1] if best else None
            try:
                duration = FFmpegWrapper.get_duration(result["audio_path"])
            except Exception:
                duration = result["duration"]

            if self.is_swallowed(text, duration, lang):
                score = -1.0
                problem = f"SWALLOWED take ({duration:.2f}s for '{text[:40]}')"
            else:
                # Interjections ("Wow.", "O!") are spelled too freely by the ASR to compare
                check = self._intelligibility(text, result["audio_path"], lang) if count_syllables(text, lang) >= 4 else None
                if check is not None:
                    result = dict(result, similarity=round(check[0], 2), heard=check[1])
                if check is None or check[0] >= TAKE_MIN_SIMILARITY:
                    return result, duration, False
                score = check[0]
                problem = f"UNINTELLIGIBLE take (similarity {check[0]:.2f}, heard '{check[1][:60]}')"

            if retry < SWALLOW_RETRIES:
                logger.warning(
                    f"[ID: {idx}] {problem}, synthesizing again with another voice reference ({retry + 1}/{SWALLOW_RETRIES})"
                )
            else:
                logger.warning(f"[ID: {idx}] {problem}, no voice reference left")
            # Skip what was used: the line clip as a whole, any other reference by its path
            skip.append("DYNAMIC" if result.get("voice_type") == "DYNAMIC" else result.get("ref") or "")
            if best is None or score > best[0]:
                # The take file is rewritten by the next call; keep the least broken one as the last resort
                kept = os.path.join(self.temp_dir, f"broken_{lang}_{idx}_{retry}.wav")
                shutil.copy(result["audio_path"], kept)
                best = (score, (dict(result, audio_path=kept), duration, True))
        return best[1]

    @staticmethod
    def _miss(estimate: float, speech: float, limit: float) -> float:
        """How far a length is from the timing; running past the limit weighs three times as much."""
        if estimate > limit:
            return (limit - speech) + 3 * (estimate - limit)
        return abs(estimate - speech)

    @classmethod
    def _rank_variants(cls, options, speech: float, limit: float):
        """Orders (text, estimate) pairs, content first (Spec 002, 2.2 step 2).

        A variant fits when duration control can bring it into the limit (estimate <= limit x TTS_MAX_SPEEDUP). When
        the main translation fits, nothing shorter than it is picked; among it and the longer variants the closest to
        the target wins. Only when it does not fit, the longest variant that fits goes first (shortest if none does).
        """
        main_est = options[0][1]
        ceiling = limit * TTS_MAX_SPEEDUP

        def key(option):
            text, est = option
            fits = est <= ceiling
            if main_est <= ceiling:
                # Full content fits: shorter variants only as a fallback after a rejected take
                return (0 if fits and est >= main_est else 1, cls._miss(est, speech, limit))
            # Main does not fit: as much content as fits, then the shortest
            return (0, -est) if fits else (1, est)

        return sorted(options, key=key)

    def _estimate_variants(self, segment: Dict, lang: str, vocals_path: str, speech: float, limit: float):
        """[(text, estimated seconds)] of the draft's length variants, best fit first; [] without variants."""
        options = [t for t in segment.get("variants") or [] if t and t.strip()]
        if len(options) < 2:
            return []
        item = {
            "index": segment["index"],
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "source_text": segment.get("text_en", ""),
        }
        try:
            estimates = self.tts.estimate_durations(item, lang, vocals_path, options)
        except Exception as e:
            logger.warning(f"[ID: {segment['index']}] Variant estimate failed: {e}")
            return []
        if not estimates or len(estimates) != len(options):
            return []
        ranked = self._rank_variants(list(zip(options, estimates)), speech, limit)
        logger.info(
            f"[ID: {segment['index']}] Variants (est.): "
            + ", ".join(f"{e:.2f}s" for _, e in zip(options, estimates))
            + f" -> picked {ranked[0][1]:.2f}s (target {speech:.2f}s, limit {limit:.2f}s)"
        )
        return ranked

    def process_segment(
        self, segment: Dict, lang: str, vocals_path: str, full_script: List[Dict], global_context: Dict, attempt_limit: int = 3
    ) -> Optional[Dict]:
        """
        Orchestrates the feedback loop for a single segment (Spec 002).

        The target is the time the original speaks (end - start); the limit adds the silence after the line
        ("max_end") and the short silence before it ("min_start"). A take is accepted when it ends within the limit and fills at least MIN_FILL_RATIO of the target.
        """
        original_text = segment.get("text_en", "")
        current_text = segment.get("text", original_text)
        speech_dur = segment["end"] - segment["start"]
        # The limit: silence after the line, plus (for a take that is still too long) a little silence before it
        target_dur = segment.get("max_end", segment["end"]) - segment.get("min_start", segment["start"])
        speaker = segment["speaker"]
        idx = segment["index"]

        attempts = []
        # Pick the length variant that fits best before synthesizing anything; the rest stay for retries
        variants = self._estimate_variants(segment, lang, vocals_path, speech_dur, target_dur)
        if variants:
            current_text = variants[0][0]
        unused = {text: est for text, est in variants[1:]}
        current_est = variants[0][1] if variants else None

        for attempt in range(1, attempt_limit + 1):
            logger.info(
                f"[ID: {idx}] Refinement Attempt {attempt}/{attempt_limit} (Target: {speech_dur:.2f}s, Limit: {target_dur:.2f}s)"
            )

            tts_item = {
                "index": idx,
                "text": current_text,
                "source_text": original_text,
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
                "max_end": segment.get("max_end", segment["end"]),
                "min_start": segment.get("min_start", segment["start"]),
            }

            synthesized = self._synthesize_sound(tts_item, lang, vocals_path, full_script)
            if not synthesized:
                logger.warning(f"[ID: {idx}] Synthesis failed on attempt {attempt}.")
                break
            result, actual_dur, broken = synthesized
            raw_audio_path = result["audio_path"]

            delta = actual_dur - target_dur
            too_long = actual_dur > target_dur * MAX_ACCEPT_SPEEDUP + 0.05
            too_short = actual_dur < speech_dur * MIN_FILL_RATIO

            # The accepted attempt is mixed much later, so it must not be shared between languages
            attempt_path = os.path.join(self.temp_dir, f"attempt_{lang}_{idx}_{attempt}.wav")
            shutil.copy(raw_audio_path, attempt_path)

            attempts.append(
                {
                    "audio_path": attempt_path,
                    "text": current_text,
                    "duration": actual_dur,
                    "delta": delta,
                    "broken": broken,
                    "voice_type": result.get("voice_type"),
                    "similarity": result.get("similarity"),
                    "heard": result.get("heard"),
                }
            )

            if not too_long and not too_short and not broken:
                logger.info(f"[ID: {idx}] ACCEPTED (Duration: {actual_dur:.2f}s, Target: {speech_dur:.2f}s)")
                self._cleanup(attempts, attempt_path)
                return {
                    "audio_path": attempt_path,
                    "final_text": current_text,
                    "duration": actual_dur,
                    "status": "ACCEPTED",
                    **self._report_fields(attempts[-1], len(attempts)),
                }

            if attempt >= attempt_limit:
                break
            if broken:
                # Every voice reference came out broken; keep what we have rather than rewording blindly
                logger.warning(f"[ID: {idx}] REJECTED (Broken on every voice reference: {actual_dur:.2f}s).")
                break
            if too_long:
                logger.info(f"[ID: {idx}] REJECTED (Too long: {actual_dur:.2f}s > {target_dur:.2f}s). Refining text...")
                aim = target_dur
            else:
                logger.info(f"[ID: {idx}] REJECTED (Too short: {actual_dur:.2f}s < {speech_dur:.2f}s). Expanding text...")
                aim = speech_dur

            # An unused variant that moves the right way is cheaper and safer than a rewrite
            alternative = self._next_variant(unused, current_est, actual_dur, too_long)
            if alternative:
                logger.info(
                    f"[ID: {idx}] Trying the {'shorter' if too_long else 'longer'} variant ({unused[alternative]:.2f}s est.)"
                )
                current_est = unused.pop(alternative)
                current_text = alternative
                continue

            context_before, context_after = self._neighbours(full_script, idx, lang)
            new_text = self.llm.refine_translation_by_duration(
                original_text=original_text,
                current_text=current_text,
                actual_dur=actual_dur,
                target_dur=aim,
                glossary=global_context.get("glossary", {}),
                context_before=context_before,
                context_after=context_after,
                language=lang,
                speaker=segment.get("speaker_label", ""),
                media=global_context.get("media", ""),
            )

            if new_text == current_text:
                if not too_long:
                    # Nothing to add: a short original ("Wow!") stays short
                    logger.info(f"[ID: {idx}] ACCEPTED short take, the LLM has nothing to add ({actual_dur:.2f}s)")
                    self._cleanup(attempts, attempt_path)
                    return {
                        "audio_path": attempt_path,
                        "final_text": current_text,
                        "duration": actual_dur,
                        "status": "ACCEPTED",
                        **self._report_fields(attempts[-1], len(attempts)),
                    }
                logger.info(f"[ID: {idx}] LLM made no changes. Stopping early.")
                break

            current_text = new_text
            current_est = None

        if not attempts:
            logger.error(f"[ID: {idx}] All synthesis attempts failed. Using silent fallback.")
            fallback_path = os.path.join(self.temp_dir, f"silent_{lang}_{idx}.wav")
            sample_rate = 24000
            with wave.open(fallback_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(b"\x00\x00" * max(1, round(speech_dur * sample_rate)))
            return {
                "audio_path": fallback_path,
                "final_text": segment.get("text_en", "..."),
                "duration": speech_dur,
                "status": "FAILED_FALLBACK",
            }

        # Final selection: sound takes first; among those that fit the limit, the one closest to the target
        sound = [a for a in attempts if not a["broken"]] or attempts
        fitting = [a for a in sound if a["duration"] <= target_dur * MAX_ACCEPT_SPEEDUP + 0.1]
        if fitting:
            best_attempt = min(fitting, key=lambda a: abs(a["duration"] - speech_dur))
            status = "VALID_FALLBACK"
        else:
            best_attempt = min(sound, key=lambda a: a["duration"])
            status = "FORCED_SHORT_FALLBACK"

        logger.warning(
            f"[ID: {idx}] {status}: Selection duration {best_attempt['duration']:.2f}s vs Target {speech_dur:.2f}s "
            f"(limit {target_dur:.2f}s)"
        )
        self._cleanup(attempts, best_attempt["audio_path"])

        return {
            "audio_path": best_attempt["audio_path"],
            "final_text": best_attempt["text"],
            "duration": best_attempt["duration"],
            "status": status,
            **self._report_fields(best_attempt, len(attempts)),
        }

    @staticmethod
    def _report_fields(attempt: Dict, count: int) -> Dict:
        """What the per-line report (DEBUG) shows about the chosen take."""
        return {
            "voice_type": attempt.get("voice_type"),
            "similarity": attempt.get("similarity"),
            "heard": attempt.get("heard"),
            "attempts": count,
        }

    @staticmethod
    def _next_variant(unused: Dict[str, float], current_est: Optional[float], actual: float, too_long: bool):
        """The unused variant estimated shorter (too long) or longer (too short) than the current text."""
        if not unused or current_est is None:
            return None
        if too_long:
            shorter = {t: e for t, e in unused.items() if e < current_est}
            return max(shorter, key=shorter.get) if shorter else None
        longer = {t: e for t, e in unused.items() if e > current_est}
        return min(longer, key=longer.get) if longer else None

    @staticmethod
    def _neighbours(full_script: List[Dict], idx: int, lang: str = ""):
        """Previous and next lines as context for the refinement: source text plus translation when known."""

        def describe(line):
            text = line.get("text_en", "")
            translated = (line.get("translations") or {}).get(lang)
            return f"{text} => {translated}" if translated else text

        before = describe(full_script[idx - 1]) if 0 < idx <= len(full_script) else ""
        after = describe(full_script[idx + 1]) if 0 <= idx < len(full_script) - 1 else ""
        return before, after

    def _cleanup(self, attempts: List[Dict], keep: str):
        if self.debug_mode:
            return
        for a in attempts:
            if a["audio_path"] != keep and os.path.exists(a["audio_path"]):
                os.remove(a["audio_path"])
