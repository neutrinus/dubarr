import logging
import threading
import gc
from typing import List, Dict, Optional, Tuple
from config import WHISPER_MODEL, MOCK_MODE, TAKE_CHECK_DEVICE, TAKE_CHECK_THREADS, ASR_GAP_PASS, ASR_GAP_MIN
from core.gpu_manager import GPUManager

try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel, decode_audio
except ImportError:
    WhisperModel = None
    decode_audio = None

SAMPLE_RATE = 16000
# Free VRAM a GPU needs before TAKE_CHECK_DEVICE=auto puts the take-check model on it
TAKE_CHECK_GPU_MB = 8500
# What Whisper writes over noise or music when nobody speaks; never kept from the gap pass
HALLUCINATIONS = {
    "you",
    "thankyou",
    "thankyouforwatching",
    "thanksforwatching",
    "pleasesubscribe",
    "instruction",
    "music",
    "applause",
    "bye",
}

logger = logging.getLogger(__name__)


def _word_bounds(segment) -> Tuple[float, float]:
    """Returns (start, end) of a segment tightened to its first and last word, if available."""
    words = getattr(segment, "words", None) or []
    if not words:
        return segment.start, segment.end
    start, end = words[0].start, words[-1].end
    if end <= start:
        return segment.start, segment.end
    return start, end


def _gaps(segments: List[Dict], total: float, min_gap: float) -> List[Tuple[float, float]]:
    """Spans of at least ``min_gap`` seconds that no segment covers."""
    gaps, cursor = [], 0.0
    for seg in sorted(segments, key=lambda x: x["start"]):
        if seg["start"] - cursor >= min_gap:
            gaps.append((cursor, seg["start"]))
        cursor = max(cursor, seg["end"])
    if total - cursor >= min_gap:
        gaps.append((cursor, total))
    return gaps


def _min_word_prob(segment) -> Optional[float]:
    """Probability of the segment's least certain word (Spec 001, 2.2c); None without word timestamps."""
    words = getattr(segment, "words", None) or []
    probs = [w.probability for w in words if getattr(w, "probability", None) is not None]
    return round(min(probs), 3) if probs else None


def _plausible(segment) -> bool:
    """A gap-pass segment Whisper is sure of: real speech, not a hallucination over noise."""
    letters = "".join(ch for ch in segment.text.lower() if ch.isalnum())
    if not letters or letters in HALLUCINATIONS:
        return False
    if segment.no_speech_prob >= 0.5 or segment.avg_logprob < -0.8:
        return False
    if getattr(segment, "compression_ratio", 1.0) > 2.4:  # "Go! Go! Go! Go! ..." loops
        return False
    words = getattr(segment, "words", None) or []
    if words and sum(w.probability for w in words) / len(words) < 0.5:
        return False
    return True


class WhisperManager:
    def __init__(
        self,
        inference_lock: Optional[threading.Lock],
    ):
        self.device = "cpu"
        self.inference_lock = inference_lock
        self.model = None
        self.status = "IDLE"  # IDLE, LOADING, READY, ERROR
        # Separate model for the Stage 5 take check, loaded on first use and released by shutdown_take_check()
        self.take_model = None
        self.take_device = "cpu"
        self._take_lock = threading.Lock()
        self.language = None  # detected by the last transcription; the second hearing reuses it

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
        # word_timestamps: cross-attention DTW alignment (same idea as whisper.cpp --dtw),
        # used to snap segment boundaries to the first/last spoken word instead of the
        # coarse segment timestamp tokens, which often include leading/trailing silence.
        # vad_filter: skip non-speech so Whisper does not hallucinate text/timestamps in silence.
        ts, info = self.model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
        logger.info(f"Whisper: Starting transcription of {info.duration:.2f}s audio (Language: {info.language})")
        self.language = info.language

        res = []
        for x in ts:
            if x.avg_logprob >= -1.0:
                start, end = _word_bounds(x)
                segment = {
                    "start": start,
                    "end": end,
                    "text": x.text.strip(),
                    "avg_logprob": x.avg_logprob,
                    "no_speech_prob": x.no_speech_prob,
                    "min_word_prob": _min_word_prob(x),
                }
                res.append(segment)
                # Log progress every few segments or every 30 seconds of audio
                if len(res) % 20 == 0:
                    logger.info(f"Whisper Progress: {x.end:.2f}s / {info.duration:.2f}s transcribed...")

        logger.info(f"Whisper: Completed transcription. Found {len(res)} segments.")
        if ASR_GAP_PASS:
            res = self._fill_gaps(audio_path, res, info.duration, info.language)
        return res

    def _fill_gaps(self, audio_path: str, res: List[Dict], total: float, language: Optional[str]) -> List[Dict]:
        """Transcribes the gaps of the first pass again, without VAD, and adds the confident segments.

        VAD drops speech under loud noise (commentary over a fight, a referee's countdown); without it Whisper hears
        it, but also hallucinates over music, hence the strict filter in ``_plausible``.
        """
        gaps = _gaps(res, total, ASR_GAP_MIN)
        if not gaps or decode_audio is None:
            return res
        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        added = []
        for g_start, g_end in gaps:
            clip = audio[int(g_start * SAMPLE_RATE) : int(g_end * SAMPLE_RATE)]
            if len(clip) < SAMPLE_RATE // 2:
                continue
            segments, _ = self.model.transcribe(
                clip, language=language, word_timestamps=True, vad_filter=False, condition_on_previous_text=False
            )
            for x in segments:
                if not _plausible(x):
                    continue
                start, end = _word_bounds(x)
                segment = {
                    "start": g_start + start,
                    "end": min(g_start + end, g_end),
                    "text": x.text.strip(),
                    "avg_logprob": x.avg_logprob,
                    "no_speech_prob": x.no_speech_prob,
                    "min_word_prob": _min_word_prob(x),
                }
                logger.info(f"Whisper gap pass: +[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")
                added.append(segment)
        logger.info(f"Whisper gap pass: {len(added)} segments added from {len(gaps)} gaps.")
        return sorted(res + added, key=lambda x: x["start"])

    def transcribe_samples(self, samples, prompt: Optional[str] = None, language: Optional[str] = None) -> str:
        """Text of a short 16 kHz clip (the second hearing of Spec 001, 2.2c); "" when nothing is heard."""
        if MOCK_MODE:
            return ""
        if not self.model:
            self.load_model()
            if not self.model:
                return ""
        try:
            segments, _ = self.model.transcribe(
                samples,
                language=language or self.language,
                initial_prompt=prompt,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(x.text.strip() for x in segments if x.no_speech_prob < 0.6).strip()
        except Exception as e:
            logger.warning(f"Whisper second hearing failed: {e}")
            return ""

    def _load_take_model(self):
        """Loads the take-check model on the CPU (default), or with TAKE_CHECK_DEVICE=auto on a GPU with room to spare."""
        device = "cpu"
        if TAKE_CHECK_DEVICE != "cpu":
            # The model (~2.5 GB) plus the headroom OmniVoice needs for long takes on the same card
            device = GPUManager.get_best_gpu(needed_mb=TAKE_CHECK_GPU_MB, purpose="Take check", timeout=0, fallback="cpu")
        device_type = "cuda" if "cuda" in device else "cpu"
        device_index = int(device.split(":")[-1]) if device_type == "cuda" else 0
        compute_type = "float16" if device_type == "cuda" else "int8"
        threads = TAKE_CHECK_THREADS if device_type == "cpu" else 0
        logging.info(
            f"Whisper: Loading take-check model '{WHISPER_MODEL}' on {device}"
            + (f" ({threads} threads)..." if threads else "...")
        )
        self.take_model = WhisperModel(
            WHISPER_MODEL, device=device_type, device_index=device_index, compute_type=compute_type, cpu_threads=threads
        )
        self.take_device = device

    def transcribe_take(self, audio_path: str, lang: str) -> Optional[str]:
        """What Whisper hears in a synthesized take, in the target language; None when it cannot check.

        No VAD (a take is speech by design) and no conditioning on previous text, so a broken take is not "repaired"
        into plausible words. Thread-safe: the Stage 5 orchestrator threads share one model.
        """
        if MOCK_MODE or WhisperModel is None:
            return None
        with self._take_lock:
            if self.take_model is None:
                try:
                    self._load_take_model()
                except Exception as e:
                    logging.error(f"Whisper: Take-check model failed to load: {e}")
                    return None
            segments, _ = self.take_model.transcribe(
                audio_path, language=lang, vad_filter=False, condition_on_previous_text=False, beam_size=1
            )
            return " ".join(s.text.strip() for s in segments).strip()

    def shutdown_take_check(self):
        """Releases the take-check model after Stage 5."""
        with self._take_lock:
            if self.take_model:
                del self.take_model
                self.take_model = None
                gc.collect()
                if torch and "cuda" in self.take_device:
                    torch.cuda.empty_cache()

    def shutdown(self):
        """Explicitly releases the model from VRAM."""
        if self.model:
            del self.model
            self.model = None
            self.status = "IDLE"
            gc.collect()
            if torch and "cuda" in self.device:
                torch.cuda.empty_cache()
