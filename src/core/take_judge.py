"""The take judge (Spec 002, 2.2 step 4): scores a synthesized take without Whisper, cheapest layer first.

1. missing or swallowed (in the synchronizer: no audio, or far too short for its text),
2. artifacts on the signal: squeaks, clipping, a long silence inside the take,
3. voice match: the take's voiceprint against the voice it cloned,
4. text match: what a CTC speech model of the target language reads in the take, compared with the text letter by
   letter on the scale of the Whisper similarity.

Layers 3 and 4 need their models; when a model is missing (MOCK_MODE, the light dev environment, a failed download)
the layer is skipped and the take is judged by the others.
"""

import logging
import os
import subprocess
import tempfile
import threading
import wave
from typing import Dict, List, Optional, Tuple

from config import MOCK_MODE, HF_TOKEN, JUDGE_ALIGN_MODEL, JUDGE_DEVICE, JUDGE_HF_MAX_DB, JUDGE_MIN_TEXT, JUDGE_MIN_VOICE
from utils import count_syllables, text_similarity

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

# Default CTC model per target language; others use MMS with the language's adapter
ALIGN_MODELS = {"pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish"}
MMS_MODEL = "facebook/mms-1b-all"
MMS_LANGS = {"pl": "pol", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "ru": "rus", "uk": "ukr",
             "cs": "ces", "sk": "slk", "nl": "nld", "sv": "swe", "en": "eng", "ja": "jpn", "zh": "cmn", "ko": "kor"}  # fmt: skip
VOICE_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

HF_SPLIT_HZ = 3500.0  # squeaks live above this; speech energy mostly below
HF_MIN_SHARE = 0.35  # share of a frame's energy above HF_SPLIT_HZ in a squeak
HF_MIN_SECONDS = 0.3
CLIP_MAX_SHARE = 0.001  # samples at full scale
GAP_MAX_SECONDS = 1.2  # silence inside a take
SILENCE_DB = -45.0


def read_wav_mono(path: str) -> Optional[Tuple["np.ndarray", int]]:
    """16-bit PCM WAV as float samples in [-1, 1] (channels averaged) and the rate; None for other formats."""
    if np is None:
        return None
    try:
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() != 2:
                return None
            rate, channels = wf.getframerate(), wf.getnchannels()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, OSError, EOFError):
        return None
    if channels > 1:
        data = data[: len(data) // channels * channels].reshape(-1, channels).mean(axis=1)
    return data, rate


def find_artifacts(samples: "np.ndarray", rate: int) -> Optional[str]:
    """What is wrong with the signal of a take (squeak, clipping, gap), or None (Spec 002, 2.2 step 4, layer 2)."""
    if np is None or samples is None or len(samples) < rate // 10:
        return None
    clipped = float(np.mean(np.abs(samples) >= 0.999))
    if clipped > CLIP_MAX_SHARE:
        return f"CLIPPING ({clipped:.1%} of samples at full scale)"

    frame = 1024 if rate >= 32000 else 512
    hop = frame // 2
    count = 1 + (len(samples) - frame) // hop
    if count < 2:
        return None
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[idx] * np.hanning(frame)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    total = power.sum(axis=1) + 1e-12
    high = power[:, freqs >= HF_SPLIT_HZ].sum(axis=1) + 1e-12
    hop_s = hop / rate

    # A squeak: most of the energy high up, far above the take's usual high band, held for a while
    high_db = 10 * np.log10(high)
    voiced = 10 * np.log10(total) > (10 * np.log10(total.max()) - 40)
    usual = float(np.median(high_db[voiced])) if voiced.any() else float(np.median(high_db))
    squeak = (high / total > HF_MIN_SHARE) & (high_db > usual + JUDGE_HF_MAX_DB)
    run = _longest_run(squeak)
    if run * hop_s >= HF_MIN_SECONDS:
        return f"SQUEAK ({run * hop_s:.2f}s of high-pitched noise)"

    # A long silence inside the take (the model lost its way); leading and trailing silence do not count
    rms_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    loud = np.nonzero(rms_db > SILENCE_DB)[0]
    if len(loud) >= 2:
        inner = rms_db[loud[0] : loud[-1] + 1] <= SILENCE_DB
        gap = _longest_run(inner) * hop_s
        if gap > GAP_MAX_SECONDS:
            return f"GAP ({gap:.2f}s of silence inside the take)"
    return None


def _longest_run(mask) -> int:
    best = cur = 0
    for flag in mask:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def ctc_greedy(log_probs, blank: int) -> List[int]:
    """The CTC model's own reading of the audio (best token per frame, repeats merged, blanks dropped)."""
    ids, prev = [], None
    for token in log_probs.argmax(-1).tolist():
        if token != prev and token != blank:
            ids.append(token)
        prev = token
    return ids


class TakeJudge:
    """Thread-safe: the Stage 5 orchestrator threads share one judge; models load on first use."""

    def __init__(self, temp_dir: str = "", signal_only: bool = False):
        self.temp_dir = temp_dir
        # TAKE_CHECK=off: only the signal checks (artifacts), no models
        self.signal_only = signal_only
        self._lock = threading.Lock()
        self._align: Dict[str, Optional[tuple]] = {}  # lang -> (model, processor) or None when unavailable
        self._voice = None  # pyannote Inference, False when unavailable
        self._prints: Dict[str, Optional["np.ndarray"]] = {}
        self._print_failed = False

    # --- layer 3: voice match ---

    def _voice_model(self):
        if self._voice is None:
            self._voice = False
            if MOCK_MODE:
                return None
            try:
                from pyannote.audio import Inference, Model

                model = Model.from_pretrained(VOICE_MODEL, token=HF_TOKEN)
                self._voice = Inference(model, window="whole")
                logger.info(f"Take judge: voiceprint model '{VOICE_MODEL}' loaded")
            except Exception as e:
                logger.warning(f"Take judge: no voiceprint model ({e}); the voice match is skipped")
        return self._voice or None

    def voiceprint(self, path: str) -> Optional["np.ndarray"]:
        model = self._voice_model()
        if model is None or np is None or not path or not os.path.exists(path):
            return None
        try:
            import torch

            # In memory, like the diarization: pyannote decoding a file path needs torchcodec, which breaks across
            # torch/ffmpeg versions (no take got a voice match that way)
            samples = self._samples_16k(path)
            if samples is None or len(samples) == 0:
                return None
            waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
            emb = np.asarray(model({"waveform": waveform, "sample_rate": 16000}), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(emb))
            return emb / norm if norm > 0 else None
        except Exception as e:
            if not self._print_failed:
                logger.warning(f"Take judge: voiceprint failed ({e}); the voice match is skipped where it fails")
                self._print_failed = True
            else:
                logger.debug(f"Take judge: voiceprint of {path} failed: {e}")
            return None

    def voice_match(self, take_path: str, ref_path: Optional[str]) -> Optional[float]:
        """Cosine similarity of the take's voice and the voice it cloned; None when it cannot tell."""
        if not ref_path:
            return None
        with self._lock:
            if ref_path not in self._prints:
                self._prints[ref_path] = self.voiceprint(ref_path)
            ref = self._prints[ref_path]
            if ref is None:
                return None
            take = self.voiceprint(take_path)
        return None if take is None else float(np.dot(ref, take))

    # --- layer 4: text match ---

    def _align_model(self, lang: str):
        if lang in self._align:
            return self._align[lang]
        self._align[lang] = None
        if MOCK_MODE:
            return None
        names = [JUDGE_ALIGN_MODEL or ALIGN_MODELS.get(lang) or MMS_MODEL]
        if names[0] != MMS_MODEL:
            names.append(MMS_MODEL)  # the multilingual model when the language's own does not load
        for name in names:
            try:
                self._align[lang] = self._load_ctc(name, lang)
                logger.info(f"Take judge: CTC model '{name}' loaded for {lang} on {JUDGE_DEVICE}")
                break
            except Exception as e:
                logger.warning(f"Take judge: CTC model '{name}' did not load for {lang}: {e}")
        if self._align[lang] is None:
            logger.error(f"Take judge: no CTC model for {lang}; takes are not checked against their text")
        return self._align[lang]

    @staticmethod
    def _load_ctc(name: str, lang: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        model_kwargs, proc_kwargs = {}, {}
        if name == MMS_MODEL:
            proc_kwargs = {"target_lang": MMS_LANGS.get(lang, lang)}
            model_kwargs = dict(proc_kwargs, ignore_mismatched_sizes=True)
        # Wav2Vec2Processor, not AutoProcessor: a model with a language model (jonatasgrosman's Polish one) makes
        # AutoProcessor pick Wav2Vec2ProcessorWithLM, which needs pyctcdecode; the CTC score needs no language model
        processor = Wav2Vec2Processor.from_pretrained(name, token=HF_TOKEN, **proc_kwargs)
        model = Wav2Vec2ForCTC.from_pretrained(name, token=HF_TOKEN, **model_kwargs).eval()
        if JUDGE_DEVICE != "cpu":
            model = model.to(JUDGE_DEVICE)
        return model, processor

    def _samples_16k(self, path: str):
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir or None)
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", tmp],
                check=True,
                capture_output=True,
                timeout=60,
            )
            read = read_wav_mono(tmp)
            return read[0] if read else None
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def text_match(self, take_path: str, text: str, lang: str) -> Optional[float]:
        """How well the text fits the take (0..1); None when it cannot tell."""
        with self._lock:
            loaded = self._align_model(lang)
            if loaded is None:
                return None
            model, processor = loaded
            try:
                import torch

                samples = self._samples_16k(take_path)
                if samples is None or len(samples) == 0:
                    return 0.0
                vocab = processor.tokenizer.get_vocab()
                # Only letters the model can write: digits are spoken as words the text does not spell
                expected = "".join(ch if (ch in vocab or ch == " ") else " " for ch in text.lower())
                inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    logits = model(inputs.input_values.to(model.device)).logits[0].float().cpu()
                ids = ctc_greedy(logits, processor.tokenizer.pad_token_id)
                # A likelihood of the text itself (exp(-CTC loss)) depends on how sure the model is of the voice:
                # clear Polish from a cloned shouting commentator scored 0.0 there; what it reads does not
                heard = processor.tokenizer.decode(ids, group_tokens=False)
                return text_similarity(expected, heard)
            except Exception as e:
                logger.warning(f"Take judge: text match failed: {e}")
                return None

    # --- all layers ---

    def judge(self, text: str, take_path: str, lang: str, ref_path: Optional[str] = None) -> Dict:
        """{"score", "problem", "voice", "text_match"}; score 0..1, problem None for a sound take.

        The score is the text match (1.0 for lines it skips), halved when the voice does not match; a take with an
        artifact scores 0.
        """
        out: Dict = {"score": 1.0, "problem": None, "voice": None, "text_match": None}
        read = read_wav_mono(take_path)
        artifact = find_artifacts(*read) if read else None
        if artifact:
            return dict(out, score=0.0, problem=artifact)
        if self.signal_only:
            return out
        voice = self.voice_match(take_path, ref_path)
        out["voice"] = None if voice is None else round(voice, 2)
        if count_syllables(text, lang) >= 4:  # interjections ("Wow!") are too short to score
            match = self.text_match(take_path, text, lang)
            out["text_match"] = None if match is None else round(match, 2)
        score = out["text_match"] if out["text_match"] is not None else 1.0
        if voice is not None and voice < JUDGE_MIN_VOICE:
            out["problem"] = f"WRONG VOICE (voice match {voice:.2f})"
            score *= 0.5
        elif out["text_match"] is not None and out["text_match"] < JUDGE_MIN_TEXT:
            out["problem"] = f"UNINTELLIGIBLE take (text match {out['text_match']:.2f})"
        out["score"] = round(score, 2)
        return out

    def shutdown(self):
        with self._lock:
            self._align.clear()
            self._voice = None
            self._prints.clear()
