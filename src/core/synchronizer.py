import logging
import os
import shutil
import wave
from typing import Dict, List, Optional
from infrastructure.ffmpeg import FFmpegWrapper
from config import MAX_ACCEPT_SPEEDUP, MIN_FILL_RATIO, MAX_PLAUSIBLE_SYLLABLES_PER_SECOND
from utils import count_syllables

logger = logging.getLogger(__name__)


# Extra syntheses of the same text after a swallowed take, each with another voice reference
SWALLOW_RETRIES = 2


class SegmentSynchronizer:
    def __init__(self, llm_manager, tts_manager, temp_dir, debug_mode=False):
        self.llm = llm_manager
        self.tts = tts_manager
        self.temp_dir = temp_dir
        self.debug_mode = debug_mode

    @staticmethod
    def is_swallowed(text: str, duration: float, lang: str) -> bool:
        """True when a take is far shorter than its text can be spoken: the TTS produced mostly silence."""
        syllables = count_syllables(text, lang)
        return syllables >= 3 and duration * MAX_PLAUSIBLE_SYLLABLES_PER_SECOND < syllables

    def _synthesize_sound(self, tts_item: Dict, lang: str, vocals_path: str, full_script: List[Dict]):
        """Synthesizes a take and re-synthesizes it with other references while it comes out swallowed.

        Returns (result, duration, swallowed) or None when synthesis failed.
        """
        skip = []
        last = None
        for retry in range(SWALLOW_RETRIES + 1):
            item = dict(tts_item, skip_refs=list(skip))
            result = self.tts.synthesize_sync(item, lang, vocals_path, full_script)
            if not result:
                return last
            try:
                duration = FFmpegWrapper.get_duration(result["audio_path"])
            except Exception:
                duration = result["duration"]
            swallowed = self.is_swallowed(tts_item["text"], duration, lang)
            last = (result, duration, swallowed)
            if not swallowed:
                return last
            logger.warning(
                f"[ID: {tts_item['index']}] SWALLOWED take ({duration:.2f}s for '{tts_item['text'][:40]}'), "
                f"synthesizing again with another voice reference ({retry + 1}/{SWALLOW_RETRIES})"
            )
            # Skip what was used: the line clip as a whole, any other reference by its path
            skip.append("DYNAMIC" if result.get("voice_type") == "DYNAMIC" else result.get("ref") or "")
            if retry < SWALLOW_RETRIES:
                # The take file is rewritten by the next call; keep the swallowed one only as the last resort
                kept = os.path.join(self.temp_dir, f"swallowed_{lang}_{tts_item['index']}.wav")
                shutil.copy(result["audio_path"], kept)
                last = (dict(result, audio_path=kept), duration, swallowed)
        return last

    def process_segment(
        self, segment: Dict, lang: str, vocals_path: str, full_script: List[Dict], global_context: Dict, attempt_limit: int = 3
    ) -> Optional[Dict]:
        """
        Orchestrates the feedback loop for a single segment (Spec 002).

        The target is the time the original speaks (end - start); the limit adds the silence after the line
        ("max_end"). A take is accepted when it ends within the limit and fills at least MIN_FILL_RATIO of the target.
        """
        original_text = segment.get("text_en", "")
        current_text = segment.get("text", original_text)
        speech_dur = segment["end"] - segment["start"]
        target_dur = segment.get("max_end", segment["end"]) - segment["start"]
        speaker = segment["speaker"]
        idx = segment["index"]

        attempts = []

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
            }

            synthesized = self._synthesize_sound(tts_item, lang, vocals_path, full_script)
            if not synthesized:
                logger.warning(f"[ID: {idx}] Synthesis failed on attempt {attempt}.")
                break
            result, actual_dur, swallowed = synthesized
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
                    "swallowed": swallowed,
                }
            )

            if not too_long and not too_short and not swallowed:
                logger.info(f"[ID: {idx}] ACCEPTED (Duration: {actual_dur:.2f}s, Target: {speech_dur:.2f}s)")
                self._cleanup(attempts, attempt_path)
                return {"audio_path": attempt_path, "final_text": current_text, "duration": actual_dur, "status": "ACCEPTED"}

            if attempt >= attempt_limit:
                break
            if swallowed:
                # Every voice reference came out silent; keep what we have rather than rewording blindly
                logger.warning(f"[ID: {idx}] REJECTED (Swallowed on every voice reference: {actual_dur:.2f}s).")
                break
            if too_long:
                logger.info(f"[ID: {idx}] REJECTED (Too long: {actual_dur:.2f}s > {target_dur:.2f}s). Refining text...")
                aim = target_dur
            else:
                logger.info(f"[ID: {idx}] REJECTED (Too short: {actual_dur:.2f}s < {speech_dur:.2f}s). Expanding text...")
                aim = speech_dur

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
                    }
                logger.info(f"[ID: {idx}] LLM made no changes. Stopping early.")
                break

            current_text = new_text

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
        sound = [a for a in attempts if not a["swallowed"]] or attempts
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
        }

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
