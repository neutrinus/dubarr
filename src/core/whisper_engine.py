import logging
import threading
import gc
from typing import List, Dict, Optional
from config import WHISPER_MODEL, MOCK_MODE
from core.gpu_manager import GPUManager

try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class WhisperManager:
    def __init__(
        self,
        inference_lock: Optional[threading.Lock],
    ):
        self.device = "cpu"
        self.inference_lock = inference_lock
        self.model = None
        self.status = "IDLE"  # IDLE, LOADING, READY, ERROR

    def load_model(self):
        """Loads the Whisper model into VRAM or RAM. Skips in MOCK_MODE."""
        if self.status == "READY":
            return

        self.status = "LOADING"
        if MOCK_MODE:
            logging.info("Whisper: MOCK_MODE enabled. Skipping model load.")
            self.status = "READY"
            return

        if WhisperModel is None:
            logging.error("Whisper: faster_whisper library is not available.")
            self.status = "ERROR"
            return

        try:
            # Dynamic GPU allocation
            # Whisper Large-v3 needs approx 4GB
            self.device = GPUManager.get_best_gpu(needed_mb=4000, purpose="Whisper")

            logging.info(f"Whisper: Loading model '{WHISPER_MODEL}' on {self.device}...")

            device_type = "cuda" if "cuda" in self.device else "cpu"
            device_index = int(self.device.split(":")[-1]) if "cuda" in self.device else 0
            compute_type = "float16" if device_type == "cuda" else "int8"

            self.model = WhisperModel(WHISPER_MODEL, device=device_type, device_index=device_index, compute_type=compute_type)
            logging.info("Whisper: Ready.")
            self.status = "READY"
        except Exception as e:
            logging.error(f"Whisper: Load Failed: {e}")
            self.status = "ERROR"

    def transcribe(self, audio_path: str) -> List[Dict]:
        """Runs transcription using the pre-loaded model."""
        if MOCK_MODE:
            return [{"start": 0.0, "end": 5.0, "text": "Mock transcription", "avg_logprob": -0.1, "no_speech_prob": 0.01}]

        if not self.model:
            # Lazy load if not ready (fallback)
            self.load_model()
            if not self.model:
                raise RuntimeError("Whisper model failed to load!")

        if self.inference_lock:
            with self.inference_lock:
                return self._run_transcribe(audio_path)
        else:
            return self._run_transcribe(audio_path)

    def _run_transcribe(self, audio_path: str) -> List[Dict]:
        ts, info = self.model.transcribe(audio_path)
        logger.info(f"Whisper: Starting transcription of {info.duration:.2f}s audio (Language: {info.language})")

        res = []
        for x in ts:
            if x.avg_logprob >= -1.0:
                segment = {
                    "start": x.start,
                    "end": x.end,
                    "text": x.text.strip(),
                    "avg_logprob": x.avg_logprob,
                    "no_speech_prob": x.no_speech_prob,
                }
                res.append(segment)
                # Log progress every few segments or every 30 seconds of audio
                if len(res) % 20 == 0:
                    logger.info(f"Whisper Progress: {x.end:.2f}s / {info.duration:.2f}s transcribed...")

        logger.info(f"Whisper: Completed transcription. Found {len(res)} segments.")
        return res

    def shutdown(self):
        """Explicitly releases the model from VRAM."""
        if self.model:
            del self.model
            self.model = None
            self.status = "IDLE"
            gc.collect()
            if torch and "cuda" in self.device:
                torch.cuda.empty_cache()
