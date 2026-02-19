import logging
import os
import shutil
from typing import Dict, List, Optional
from infrastructure.ffmpeg import FFmpegWrapper

logger = logging.getLogger(__name__)


class SegmentSynchronizer:
    def __init__(self, llm_manager, tts_manager, temp_dir, debug_mode=False):
        self.llm = llm_manager
        self.tts = tts_manager
        self.temp_dir = temp_dir
        self.debug_mode = debug_mode

    def process_segment(
        self, segment: Dict, lang: str, vocals_path: str, full_script: List[Dict], global_context: Dict, attempt_limit: int = 3
    ) -> Optional[Dict]:
        """
        Orchestrates the feedback loop for a single segment.
        Returns the final accepted audio path and text.
        """
        original_text = segment.get("text_en", "")
        # If 'text' exists (from draft), use it. Otherwise use original.
        current_text = segment.get("text", original_text)
        target_dur = segment["end"] - segment["start"]
        speaker = segment["speaker"]
        idx = segment["index"]

        attempts = []

        for attempt in range(1, attempt_limit + 1):
            logger.info(f"[ID: {idx}] Refinement Attempt {attempt}/{attempt_limit} (Target: {target_dur:.2f}s)")

            # 1. Synthesize
            tts_item = {
                "index": idx,
                "text": current_text,
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
            }

            # Use unique path for each attempt to avoid overwriting race conditions or locks
            # The TTS Manager handles temp paths internally, but returns the path.
            # We might want to copy it to a safe place if we iterate.

            result = self.tts.synthesize_sync(tts_item, lang, vocals_path, full_script)
            if not result:
                logger.warning(f"[ID: {idx}] Synthesis failed on attempt {attempt}.")
                break

            raw_audio_path = result["audio_path"]

            # Verify duration physically
            try:
                actual_dur = FFmpegWrapper.get_duration(raw_audio_path)
            except Exception:
                actual_dur = result["duration"]  # Fallback to TTS reported duration

            # 2. Measure & Decide
            delta = actual_dur - target_dur
            abs_delta = abs(delta)

            # Acceptance Criteria:
            # - Perfect: < 0.5s diff (human reaction time)
            # - Good enough: < 15% diff (stretchable)
            # - Short segments (<2s): Strict 0.3s
            if target_dur < 2.0:
                is_acceptable = abs_delta < 0.4
            else:
                is_acceptable = abs_delta < 0.6 or (abs_delta / target_dur < 0.15)

            # Save attempt (move file to safe attempt path)
            attempt_path = os.path.join(self.temp_dir, f"attempt_{idx}_{attempt}.wav")
            shutil.copy(raw_audio_path, attempt_path)

            attempts.append(
                {
                    "audio_path": attempt_path,
                    "text": current_text,
                    "duration": actual_dur,
                    "delta": delta,
                    "score": abs_delta,  # Lower is better
                }
            )

            if is_acceptable:
                logger.info(f"[ID: {idx}] ACCEPTED (Delta: {delta:+.2f}s)")
                return {"audio_path": attempt_path, "final_text": current_text, "duration": actual_dur, "status": "ACCEPTED"}

            # 3. Refine (if not last attempt)
            if attempt < attempt_limit:
                logger.info(f"[ID: {idx}] REJECTED (Delta: {delta:+.2f}s). Refining text...")
                new_text = self.llm.refine_translation_by_duration(
                    original_text=original_text,
                    current_text=current_text,
                    actual_dur=actual_dur,
                    target_dur=target_dur,
                    glossary=global_context.get("glossary", {}),
                )

                if new_text == current_text:
                    logger.info(f"[ID: {idx}] LLM made no changes. Stopping early.")
                    break

                current_text = new_text

        # 4. Fallback: Pick best attempt
        if not attempts:
            logger.error(f"[ID: {idx}] All synthesis attempts failed.")
            return None

        best_attempt = min(attempts, key=lambda x: x["score"])
        logger.warning(
            f"[ID: {idx}] FALLBACK to best attempt {attempts.index(best_attempt) + 1} (Delta: {best_attempt['delta']:+.2f}s)"
        )

        return {
            "audio_path": best_attempt["audio_path"],
            "final_text": best_attempt["text"],
            "duration": best_attempt["duration"],
            "status": "FALLBACK",
        }
