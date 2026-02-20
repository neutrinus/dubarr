import logging
import gc
import os
import threading
from typing import List, Dict, Optional
from config import MOCK_MODE
from core.gpu_manager import GPUManager

try:
    import torch
except ImportError:
    torch = None

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None


class DiarizationManager:
    def __init__(
        self,
        inference_lock: Optional[threading.Lock],
    ):
        self.device = "cpu"
        self.inference_lock = inference_lock
        self.pipeline = None
        self.status = "IDLE"  # IDLE, LOADING, READY, ERROR

    def load_model(self):
        """Loads the Pyannote Diarization pipeline. Skips in MOCK_MODE."""
        if self.status == "READY":
            return

        self.status = "LOADING"
        if MOCK_MODE:
            logging.info("Diarization: MOCK_MODE enabled. Skipping model load.")
            self.status = "READY"
            return

        if Pipeline is None:
            logging.error("Diarization: pyannote.audio library is not available.")
            self.status = "ERROR"
            return

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logging.error("Diarization: HF_TOKEN missing. Cannot load pipeline.")
            self.status = "ERROR"
            return

        try:
            # Dynamic GPU allocation
            # Pyannote needs approx 2GB
            self.device = GPUManager.get_best_gpu(needed_mb=2000, purpose="Diarization")

            logging.info(f"Diarization: Loading pipeline on {self.device}...")
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
            if torch and "cuda" in self.device:
                self.pipeline.to(torch.device(self.device))

            logging.info("Diarization: Ready.")
            self.status = "READY"
        except Exception as e:
            logging.error(f"Diarization: Load Failed: {e}")
            self.status = "ERROR"

    def diarize(self, audio_path: str) -> List[Dict]:
        """Runs diarization using the pre-loaded pipeline."""
        if MOCK_MODE:
            return [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]

        if not self.pipeline:
            self.load_model()
            if not self.pipeline:
                raise RuntimeError("Diarization pipeline failed to load!")

        if self.inference_lock:
            with self.inference_lock:
                return self._run_diarize(audio_path)
        else:
            return self._run_diarize(audio_path)

    def _run_diarize(self, audio_path: str) -> List[Dict]:
        res = self.pipeline(audio_path)
        # Handle different return types from pyannote
        annotation = getattr(res, "speaker_diarization", getattr(res, "diarization", getattr(res, "annotation", res)))
        diar_result = [
            {"start": s.start, "end": s.end, "speaker": label} for s, _, label in annotation.itertracks(yield_label=True)
        ]
        return diar_result

    def shutdown(self):
        """Explicitly releases the pipeline from VRAM."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.status = "IDLE"
            gc.collect()
            if torch and "cuda" in self.device:
                torch.cuda.empty_cache()
