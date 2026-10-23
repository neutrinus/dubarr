import os
import logging
import time
import shutil
import threading
import gc
import subprocess
from typing import List, Dict, Optional, Any
from infrastructure.tts_client import TTSWrapper, ref_text_path, write_ref_text
from utils import measure_zcr, count_syllables
from config import MOCK_MODE, TTS_BACKEND
from core import audio as audio_processor
from core.gpu_manager import GPUManager

try:
    import torch
except ImportError:
    torch = None

# Zero-shot cloning from shorter clips sounds noticeably off (XTTS wants ~6s of clean speech)
MIN_DYNAMIC_REF_DUR = 4.0
# OmniVoice clones well from 3-10 s; a clip of the line itself carries that scene's emotion
LINE_REF_RANGE = (3.0, 10.0)


class TTSManager:
    def __init__(
        self,
        inference_lock: Optional[threading.Lock],
        temp_dir: str,
        speaker_refs: Dict,
        abort_event: threading.Event,
        service: Optional[Any] = None,  # TTSService
        backend: str = TTS_BACKEND,
    ):
        self.device = "cpu"
        self.inference_lock = inference_lock
        self.temp_dir = temp_dir
        self.speaker_refs = speaker_refs  # Golden samples
        self.abort_event = abort_event
        self.service = service
        self.backend = backend
        self.last_good_samples = {}
        self.engine = None
        self._engine_lock = threading.Lock()
        self.status = "IDLE"  # IDLE, LOADING, READY, ERROR

    def load_engine(self):
        """Initializes the TTS engine with thread safety. Skips in MOCK_MODE."""
        if self.status == "READY":
            return

        with self._engine_lock:
            if self.status == "READY":
                return

            if MOCK_MODE:
                logging.info("TTS: MOCK_MODE enabled. Skipping engine load.")
                self.status = "READY"
                return

            self.status = "LOADING"
            try:
                # Dynamic GPU allocation
                # XTTS v2 and OmniVoice (fp16) need approx 4GB
                self.device = GPUManager.get_best_gpu(needed_mb=4000, purpose=self.backend)

                gpu_id = 0
                if "cuda" in self.device:
                    gpu_id = int(self.device.split(":")[-1])

                logging.info(f"TTS: Using backend '{self.backend}'")
                self.engine = TTSWrapper(gpu_id=gpu_id, backend=self.backend)
                self.status = "READY"

            except Exception as e:
                logging.error(f"TTS: Load Failed: {e}")
                self.status = "ERROR"
                raise e

    def synthesize_sync(self, item: Dict, lang: str, vocals_path: str, script: List[Dict]) -> Optional[Dict]:
        """Synchronous synthesis without queue management."""
        self.load_engine()
        result = self._synthesize_item(item, lang, vocals_path, script)
        if result:
            return {
                "audio_path": result["payload"]["raw_path"],
                "duration": result["duration"],
                "voice_type": result["payload"]["voice_type"],
            }
        return None

    def shutdown(self):
        """Explicitly stops the TTS engine."""
        with self._engine_lock:
            if self.engine:
                del self.engine
                self.engine = None
                self.status = "IDLE"
                gc.collect()
                if torch and "cuda" in self.device:
                    torch.cuda.empty_cache()

    def _run_synthesis(self, *args, **kwargs):
        """Wrapper for TTS inference that respects the global lock or uses the RPC service."""
        if MOCK_MODE:
            # Extract output_path from args or kwargs
            # Signature: synthesize(text, ref_audio, output_path, language="en", max_duration=None)
            if len(args) >= 3:
                output_path = args[2]
            else:
                output_path = kwargs.get("output_path")

            # Generate a 1 second silent WAV file using ffmpeg
            cmd = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1", "-y", output_path]
            subprocess.run(cmd, capture_output=True, check=True)
            return

        # RPC Service Path
        if self.service:
            # We must use a direct call helper to avoid recursion if the worker
            # thread happens to call this method (which it won't, but it's safe)
            def _direct_call(*a, **kw):
                return self.engine.synthesize(*a, **kw)

            # Synthesis tasks are Priority 1 (High)
            future = self.service.submit(_direct_call, *args, priority=1, **kwargs)
            return future.result()

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

        # 2. Prepare TTS inputs
        clean_text = item["text"].strip()
        raw_path = os.path.join(self.temp_dir, f"raw_{idx}.wav")

        # 3. Calculate constraints
        slot_dur = item["end"] - item["start"]
        syl_count = count_syllables(clean_text, lang)
        max_allowed_dur = max((syl_count * 0.4) + 1.5, slot_dur + 1.5)

        try:
            if not chosen_ref:
                logging.error(f"TTS: [ID: {idx}] No reference audio found for speaker {spk}. Skipping.")
                return None

            t0 = time.perf_counter()
            try:
                self._run_synthesis(clean_text, chosen_ref, raw_path, language=lang, max_duration=slot_dur)
            except ValueError as ve:
                if "XTTS Tensor Error" in str(ve):
                    logging.warning(f"TTS: [ID: {idx}] Tensor Error with ref {chosen_ref}. Trying global fallback...")
                    # Try with first available golden ref that is NOT the current one (if possible)
                    fallback_ref = None
                    for p in self.speaker_refs.values():
                        if os.path.exists(p) and p != chosen_ref:
                            fallback_ref = p
                            break

                    if not fallback_ref:
                        for p in self.last_good_samples.values():
                            if os.path.exists(p) and p != chosen_ref:
                                fallback_ref = p
                                break

                    if fallback_ref:
                        self._run_synthesis(clean_text, fallback_ref, raw_path, language=lang, max_duration=slot_dur)
                        voice_type = "TENSOR_FALLBACK"
                    else:
                        raise ve  # Re-raise if no other fallback
                else:
                    raise ve

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

    def set_speaker_refs(self, speaker_refs: Dict):
        """Installs the golden references of the current video and drops samples left over from the previous one."""
        self.speaker_refs = dict(speaker_refs or {})
        self.last_good_samples = {}

    def _select_reference(self, item: Dict, vocals_path: str):
        """Implements the Voice Migration Strategy hierarchy.

        With XTTS the golden reference (a long, clean, scored sample of the speaker) comes first:
        it needs several seconds of clean speech, and a single reference keeps the voice consistent.
        OmniVoice clones well from 3-10 s, so a clean clip of the line itself is tried first there:
        it carries the emotion of the scene. Short lines always use the golden reference.
        """
        spk = item["speaker"]
        idx = item["index"]
        dur = item["end"] - item["start"]
        golden_ref = self.speaker_refs.get(spk)
        has_golden = bool(golden_ref and os.path.exists(golden_ref))

        if self.backend == "omnivoice":
            use_line = LINE_REF_RANGE[0] <= dur <= LINE_REF_RANGE[1]
        else:
            use_line = not has_golden and dur >= MIN_DYNAMIC_REF_DUR

        # 1. Dynamic extraction from the line itself
        if use_line:
            dyn_ref = self._extract_line_ref(item, vocals_path)
            if dyn_ref:
                return dyn_ref, "DYNAMIC"

        # 2. Golden (scored speaker reference)
        if has_golden:
            return golden_ref, "GOLDEN"

        # 3. Last Good dynamic sample of this speaker
        if spk in self.last_good_samples and os.path.exists(self.last_good_samples[spk]):
            return self.last_good_samples[spk], "LAST_GOOD"

        # 4. Global Fallback (Any Golden or Last Good)
        # If we reach here, we have no specific reference for this speaker.
        # We pick the first available one from Golden refs, then Last Good.
        all_goldens = [p for p in self.speaker_refs.values() if os.path.exists(p)]
        if all_goldens:
            logging.info(f"TTS: [ID: {idx}] Using Global Golden fallback for speaker {spk}")
            return all_goldens[0], "FALLBACK_GOLDEN"

        all_last_goods = [p for p in self.last_good_samples.values() if os.path.exists(p)]
        if all_last_goods:
            logging.info(f"TTS: [ID: {idx}] Using Global LastGood fallback for speaker {spk}")
            return all_last_goods[0], "FALLBACK_LASTGOOD"

        return None, "UNKNOWN"

    def _extract_line_ref(self, item: Dict, vocals_path: str) -> Optional[str]:
        """Cuts the line from the vocals track; returns its path if it is clean enough to clone from."""
        spk = item["speaker"]
        dyn_path = os.path.join(self.temp_dir, f"dyn_{item['index']}.wav")
        audio_processor.extract_clean_segment(vocals_path, item["start"], item["end"], dyn_path)
        zcr = measure_zcr(dyn_path)

        if os.path.exists(dyn_path) and os.path.getsize(dyn_path) > 4000 and zcr < 0.25:
            write_ref_text(dyn_path, item.get("source_text", ""))
            # Update Last Good Sample
            last_good_path = os.path.join(self.temp_dir, f"last_good_{spk}.wav")
            shutil.copy(dyn_path, last_good_path)
            if os.path.exists(ref_text_path(dyn_path)):
                shutil.copy(ref_text_path(dyn_path), ref_text_path(last_good_path))
            elif os.path.exists(ref_text_path(last_good_path)):
                os.remove(ref_text_path(last_good_path))
            self.last_good_samples[spk] = last_good_path
            return dyn_path

        if os.path.exists(dyn_path):
            os.remove(dyn_path)
        return None
