import os
import logging
import time
import shutil
import threading
import queue
import gc
import subprocess
from typing import List, Dict, Optional
from infrastructure.tts_client import F5TTSWrapper
from utils import measure_zcr, count_syllables
from config import MOCK_MODE
from core import audio as audio_processor

try:
    import torch
except ImportError:
    torch = None


class TTSManager:
    def __init__(
        self,
        device: str,
        inference_lock: Optional[threading.Lock],
        temp_dir: str,
        speaker_refs: Dict,
        abort_event: threading.Event,
    ):
        self.device = device
        self.inference_lock = inference_lock
        self.temp_dir = temp_dir
        self.speaker_refs = speaker_refs  # Golden samples
        self.abort_event = abort_event
        self.last_good_samples = {}
        self.engine = None

    def load_engine(self):
        """Initializes the TTS engine. Skips in MOCK_MODE."""
        if MOCK_MODE:
            logging.info("TTS: MOCK_MODE enabled. Skipping engine load.")
            return

        if not self.engine:
            # F5TTSWrapper expects an int gpu_id or runs on CPU if not found/configured
            gpu_id = 0
            if "cuda" in self.device:
                gpu_id = int(self.device.split(":")[-1])

            self.engine = F5TTSWrapper(gpu_id=gpu_id)

    def tts_worker(
        self,
        lang: str,
        q_in: queue.Queue,
        q_out: queue.Queue,
        translations: List,
        vocals_path: str,
        script: List[Dict],
        durations_dict: Dict,
    ):
        """Threaded worker for TTS synthesis with dynamic voice migration strategy."""
        t_tts_total = 0
        try:
            self.load_engine()

            while not self.abort_event.is_set():
                try:
                    item = q_in.get(timeout=2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                translations.append(item)

                voice_data = self._synthesize_item(item, lang, vocals_path, script)
                if voice_data:
                    q_out.put(voice_data["payload"])
                    t_tts_total += voice_data["duration"]

                q_in.task_done()

            durations_dict[f"5c. TTS Synthesis ({lang})"] = t_tts_total
        except Exception:
            logging.exception("TTS: Worker failed")
            self.abort_event.set()
        finally:
            q_out.put(None)
            if self.engine:
                del self.engine
                self.engine = None
                gc.collect()
                if torch and "cuda" in self.device:
                    torch.cuda.empty_cache()

    def _run_synthesis(self, *args, **kwargs):
        """Wrapper for TTS inference that respects the global lock if needed. Mocks in MOCK_MODE."""
        if MOCK_MODE:
            # Extract output_path from args or kwargs
            # Signature: synthesize(text, ref_audio, output_path, ref_text="", language="en")
            if len(args) >= 3:
                output_path = args[2]
            else:
                output_path = kwargs.get("output_path")

            # Generate a 1 second silent WAV file using ffmpeg
            cmd = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1", "-y", output_path]
            subprocess.run(cmd, capture_output=True, check=True)
            return

        if self.inference_lock:
            with self.inference_lock:
                return self.engine.synthesize(*args, **kwargs)
        else:
            return self.engine.synthesize(*args, **kwargs)

    def _synthesize_item(self, item: Dict, lang: str, vocals_path: str, script: List[Dict]) -> Dict:
        """Internal method to handle a single line synthesis with all fallbacks."""
        spk = item["speaker"]
        idx = item["index"]

        # 1. Select Reference Audio (Migration Strategy)
        chosen_ref, voice_type = self._select_reference(item, vocals_path)

        # 2. Prepare F5-TTS inputs
        clean_text = item["text"].strip()
        ref_text = script[idx].get("text", "")
        raw_path = os.path.join(self.temp_dir, f"raw_{idx}.wav")

        # 3. Calculate constraints
        syl_count = count_syllables(clean_text, lang)
        max_allowed_dur = max((syl_count * 0.4) + 1.5, (item["end"] - item["start"]) + 1.5)

        try:
            if not chosen_ref:
                logging.error(f"TTS: [ID: {idx}] No reference audio found for speaker {spk}. Skipping.")
                return None

            t0 = time.perf_counter()
            self._run_synthesis(clean_text, chosen_ref, raw_path, ref_text=ref_text, language=lang)
            dt = time.perf_counter() - t0

            # 4. Verify Output
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
                logging.debug(f"TTS: [ID: {idx}] Done in {dt:.2f}s ({voice_type})")
                payload = {
                    "item": item,
                    "raw_path": raw_path,
                    "max_dur": max_allowed_dur,
                    "voice_type": voice_type,
                    "lang": lang,
                }

                return {"payload": payload, "duration": dt}
            else:
                logging.warning(f"TTS: [ID: {idx}] Produced invalid/empty file.")
                return None

        except Exception as e:
            logging.error(f"TTS: [ID: {idx}] Gen failed: {e}")
            return None

    def _select_reference(self, item: Dict, vocals_path: str):
        """Implements the Voice Migration Strategy hierarchy."""
        spk = item["speaker"]
        idx = item["index"]
        start, end = item["start"], item["end"]
        dur = end - start

        # Path for dynamic extraction
        dyn_path = os.path.join(self.temp_dir, f"dyn_{idx}.wav")

        # Attempt Dynamic Extraction
        audio_processor.extract_clean_segment(vocals_path, start, end, dyn_path)
        zcr = measure_zcr(dyn_path)

        is_good_dynamic = os.path.exists(dyn_path) and os.path.getsize(dyn_path) > 4000 and zcr < 0.25 and dur > 0.8

        if is_good_dynamic:
            # Update Last Good Sample
            last_good_path = os.path.join(self.temp_dir, f"last_good_{spk}.wav")
            shutil.copy(dyn_path, last_good_path)
            self.last_good_samples[spk] = last_good_path
            return dyn_path, "DYNAMIC"

        # Fallback 1: Last Good
        if spk in self.last_good_samples and os.path.exists(self.last_good_samples[spk]):
            if os.path.exists(dyn_path):
                os.remove(dyn_path)
            return self.last_good_samples[spk], "LAST_GOOD"

        # Fallback 2: Golden (Global)
        golden_ref = self.speaker_refs.get(spk)
        if golden_ref and os.path.exists(golden_ref):
            if os.path.exists(dyn_path):
                os.remove(dyn_path)
            return golden_ref, "GOLDEN"

        if os.path.exists(dyn_path):
            os.remove(dyn_path)
        return None, "UNKNOWN"
