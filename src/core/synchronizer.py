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
        self, segment: Dict, lang: str, vocals_path: str, full_script: List[Dict], global_context: Dict, attempt_limit: int = 5
    ) -> Optional[Dict]:
        """
        Orchestrates the feedback loop for a single segment.
        Returns the final accepted audio path and text.
        New Strategy: Strictly <= target duration, 5 attempts.
        """
        original_text = segment.get("text_en", "")
        current_text = segment.get("text", original_text)
        target_dur = segment["end"] - segment["start"]
        speaker = segment["speaker"]
        idx = segment["index"]

        attempts = []

        for attempt in range(1, attempt_limit + 1):
            logger.info(f"[ID: {idx}] Refinement Attempt {attempt}/{attempt_limit} (Target: {target_dur:.2f}s)")

            tts_item = {
                "index": idx,
                "text": current_text,
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
            }

            result = self.tts.synthesize_sync(tts_item, lang, vocals_path, full_script)
            if not result:
                logger.warning(f"[ID: {idx}] Synthesis failed on attempt {attempt}.")
                break

            raw_audio_path = result["audio_path"]

            try:
                actual_dur = FFmpegWrapper.get_duration(raw_audio_path)
            except Exception:
                actual_dur = result["duration"]

            delta = actual_dur - target_dur

            # Acceptance Criteria: Strictly not longer than original
            # We allow a very tiny margin (0.05s) for technical jitter
            is_acceptable = actual_dur <= (target_dur + 0.05)

            # Save attempt
            attempt_path = os.path.join(self.temp_dir, f"attempt_{idx}_{attempt}.wav")
            shutil.copy(raw_audio_path, attempt_path)

            attempts.append(
                {
                    "audio_path": attempt_path,
                    "text": current_text,
                    "duration": actual_dur,
                    "delta": delta,
                }
            )

            if is_acceptable:
                logger.info(f"[ID: {idx}] ACCEPTED (Duration: {actual_dur:.2f}s, Target: {target_dur:.2f}s)")
                if not self.debug_mode:
                    for a in attempts:
                        if a["audio_path"] != attempt_path and os.path.exists(a["audio_path"]):
                            os.remove(a["audio_path"])
                return {"audio_path": attempt_path, "final_text": current_text, "duration": actual_dur, "status": "ACCEPTED"}

            # 3. Refine (if not last attempt)
            if attempt < attempt_limit:
                logger.info(f"[ID: {idx}] REJECTED (Too long: {actual_dur:.2f}s > {target_dur:.2f}s). Refining text...")
                
                # Get context from script
                context_before = ""
                context_after = ""
                if idx > 0:
                    context_before = full_script[idx-1].get("text_en", "")
                if idx < len(full_script) - 1:
                    context_after = full_script[idx+1].get("text_en", "")

                new_text = self.llm.refine_translation_by_duration(
                    original_text=original_text,
                    current_text=current_text,
                    actual_dur=actual_dur,
                    target_dur=target_dur,
                    glossary=global_context.get("glossary", {}),
                    context_before=context_before,
                    context_after=context_after
                )

                if new_text == current_text:
                    logger.info(f"[ID: {idx}] LLM made no changes. Stopping early.")
                    break

                current_text = new_text

        # 4. Final Selection: Closest to target BUT NOT LONGER
        if not attempts:
            logger.error(f"[ID: {idx}] All synthesis attempts failed.")
            return None

        # Filter attempts that are within time limit
        valid_attempts = [a for a in attempts if a["duration"] <= (target_dur + 0.1)]

        if valid_attempts:
            # Pick the longest one that fits (closest to original duration)
            best_attempt = max(valid_attempts, key=lambda x: x["duration"])
            status = "VALID_FALLBACK"
        else:
            # If all failed to fit, pick the shortest one available
            best_attempt = min(attempts, key=lambda x: x["duration"])
            status = "FORCED_SHORT_FALLBACK"

        logger.warning(f"[ID: {idx}] {status}: Selection duration {best_attempt['duration']:.2f}s vs Target {target_dur:.2f}s")

        # Clean up other attempts
        if not self.debug_mode:
            for a in attempts:
                if a["audio_path"] != best_attempt["audio_path"] and os.path.exists(a["audio_path"]):
                    os.remove(a["audio_path"])

        return {
            "audio_path": best_attempt["audio_path"],
            "final_text": best_attempt["text"],
            "duration": best_attempt["duration"],
            "status": status,
        }
