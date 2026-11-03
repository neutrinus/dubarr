import os
import logging
import re
import time
import shutil
import threading
import gc
import subprocess
from typing import List, Dict, Optional, Any, Tuple
from infrastructure.tts_client import TTSWrapper, ref_text_path, write_ref_text
from utils import measure_zcr, count_syllables
from config import MOCK_MODE, TTS_DEVICE, TTS_MAX_CHUNK_SECONDS, REF_MIN_CLARITY_DB, TAKE_MIN_SIMILARITY
from core import audio as audio_processor
from core.gpu_manager import GPUManager

try:
    import torch
except ImportError:
    torch = None

# Rough speaking pace used only to size synthesis chunks (Polish reads ~15 characters per second)
CHARS_PER_SECOND = 15.0
# Where an over-long sentence may be broken, in order of preference (words are the last resort)
CLAUSE_BREAK = re.compile(r"(?<=[,;:—–])\s+")


def _split_words(text: str, max_chars: int) -> List[str]:
    chunks, part = [], []
    for word in text.split():
        if part and len(" ".join(part + [word])) > max_chars:
            chunks.append(" ".join(part))
            part = []
        part.append(word)
    if part:
        chunks.append(" ".join(part))
    return chunks


def _fit_piece(piece: str, max_chars: int) -> List[str]:
    """Breaks one sentence into pieces that fit, on commas/semicolons first and on words only if unavoidable."""
    if len(piece) <= max_chars:
        return [piece]
    clauses = [p.strip() for p in CLAUSE_BREAK.split(piece) if p.strip()]
    if len(clauses) > 1:
        fitted: List[str] = []
        for clause in clauses:
            fitted.extend(_fit_piece(clause, max_chars))
        return fitted
    return _split_words(piece, max_chars)


def split_for_tts(text: str, slot_dur: float, max_chunk: float = TTS_MAX_CHUNK_SECONDS) -> List[Tuple[str, float]]:
    """Splits a long line into chunks that each fit in ``max_chunk`` seconds of speech.

    One synthesis call per sentence group keeps the dub aligned with the original pauses and caps the VRAM
    peak, which grows with the length of generated audio. Chunks break at sentence ends first, then at
    commas/semicolons, and only then mid-sentence. Short lines are returned unchanged.
    """
    text = (text or "").strip()
    if not text or slot_dur <= max_chunk:
        return [(text, slot_dur)]

    sentences = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    if len(sentences) < 2:
        sentences = [text]

    max_chars = max(int(max_chunk * CHARS_PER_SECOND), 20)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        for piece in _fit_piece(sentence, max_chars):
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)

    if len(chunks) < 2:
        return [(text, slot_dur)]

    total_chars = sum(len(c) for c in chunks)
    return [(c, max(0.8, slot_dur * len(c) / total_chars)) for c in chunks]


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
    ):
        self.device = "cpu"
        self.inference_lock = inference_lock
        self.temp_dir = temp_dir
        self.speaker_refs = {}  # speaker -> reference candidates, best first (Spec 002, 2.3)
        self.genders = {}
        self.audition = {}  # reference path -> audition score, when the voice audition ran
        self.line_refs_off = set()  # speakers whose voice clones badly; they skip line clips
        self.set_speaker_refs(speaker_refs)
        self.abort_event = abort_event
        self.service = service
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
                if TTS_DEVICE == "cpu":
                    # Keeps every GPU free for other work (e.g. a shared LLM); synthesis is many times slower
                    logging.info("TTS: TTS_DEVICE=cpu, synthesizing on the CPU.")
                    self.device = "cpu"
                    gpu_id = -1
                else:
                    # Dynamic GPU allocation
                    # OmniVoice (fp16) needs approx 4GB
                    self.device = GPUManager.get_best_gpu(needed_mb=4000, purpose="TTS (OmniVoice)")
                    gpu_id = 0
                    if "cuda" in self.device:
                        gpu_id = int(self.device.split(":")[-1])

                self.engine = TTSWrapper(gpu_id=gpu_id)
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
                "ref": result["payload"].get("ref"),
            }
        return None

    def estimate_durations(self, item: Dict, lang: str, vocals_path: str, texts: List[str]) -> Optional[List[float]]:
        """Natural length of each text with the voice this line would use (Spec 002, 2.2 step 2); None if unknown."""
        if MOCK_MODE:
            return [count_syllables(t, lang) / 5.0 for t in texts]
        self.load_engine()
        ref, _ = self._select_reference(item, vocals_path, lang)
        if not ref or not self.engine or not hasattr(self.engine, "estimate"):
            return None

        def _direct_call(t, r):
            return self.engine.estimate(t, r)

        if self.service:
            return self.service.submit(_direct_call, texts, ref, priority=1).result()
        return self.engine.estimate(texts, ref)

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

    def _synthesize_chunked(
        self, text: str, ref_audio: str, out_path: str, lang: str, slot_dur: float, idx: int, limit_dur: float = None
    ):
        """Synthesizes a line, splitting long ones so no single call generates a very long clip.

        ``slot_dur`` is how long the original speaks (the target), ``limit_dur`` the latest end including the silence
        after the line. Each chunk gets its share of both.
        """
        limit_dur = max(limit_dur or slot_dur, slot_dur)
        chunks = split_for_tts(text, slot_dur)
        if len(chunks) < 2:
            self._run_synthesis(text, ref_audio, out_path, language=lang, max_duration=limit_dur, target_duration=slot_dur)
            return

        logging.info(f"TTS: [ID: {idx}] {slot_dur:.1f}s line split into {len(chunks)} chunks")
        chunk_paths = []
        for i, (chunk_text, chunk_dur) in enumerate(chunks):
            chunk_path = os.path.join(self.temp_dir, f"chunk_{lang}_{idx}_{i}.wav")
            self._run_synthesis(
                chunk_text,
                ref_audio,
                chunk_path,
                language=lang,
                max_duration=chunk_dur * limit_dur / slot_dur,
                target_duration=chunk_dur,
            )
            chunk_paths.append(chunk_path)

        try:
            self._concat_wavs(chunk_paths, out_path)
        finally:
            for path in chunk_paths:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def _concat_wavs(paths: List[str], out_path: str):
        list_file = out_path + ".concat.txt"
        with open(list_file, "w", encoding="utf-8") as f_list:
            for path in paths:
                f_list.write(f"file '{path}'\n")
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path, "-y"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg could not join {len(paths)} synthesis chunks: {result.stderr[-300:]!r}")
        os.remove(list_file)

    def _synthesize_item(self, item: Dict, lang: str, vocals_path: str, script: List[Dict]) -> Dict:
        """Internal method to handle a single line synthesis with all fallbacks."""
        spk = item["speaker"]
        idx = item["index"]

        # 1. Select Reference Audio (Migration Strategy); a retry skips the references already tried
        chosen_ref, voice_type = self._select_reference(item, vocals_path, lang, skip=item.get("skip_refs") or ())

        # 2. Prepare TTS inputs
        clean_text = item["text"].strip()
        # Languages are produced concurrently and share segment indexes
        raw_path = os.path.join(self.temp_dir, f"raw_{lang}_{idx}.wav")

        # 3. Calculate constraints
        slot_dur = item["end"] - item["start"]
        limit_dur = item.get("max_end", item["end"]) - item.get("min_start", item["start"])
        syl_count = count_syllables(clean_text, lang)
        max_allowed_dur = max((syl_count * 0.4) + 1.5, slot_dur + 1.5)

        try:
            if not chosen_ref:
                logging.error(f"TTS: [ID: {idx}] No reference audio found for speaker {spk}. Skipping.")
                return None

            t0 = time.perf_counter()
            self._synthesize_chunked(clean_text, chosen_ref, raw_path, lang, slot_dur, idx, limit_dur)

            dt = time.perf_counter() - t0

            # 4. Verify Output
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
                logging.debug(f"TTS: [ID: {idx}] Done in {dt:.2f}s ({voice_type})")
                payload = {
                    "item": item,
                    "raw_path": raw_path,
                    "max_dur": max_allowed_dur,
                    "voice_type": voice_type,
                    "ref": chosen_ref,
                    "lang": lang,
                }

                return {"payload": payload, "duration": dt}
            else:
                logging.warning(f"TTS: [ID: {idx}] Produced invalid/empty file.")
        except Exception as e:
            logging.error(f"TTS: [ID: {idx}] Gen failed: {e}")
        # No audio, but the caller learns which reference failed and can try another one
        return {"payload": {"item": item, "raw_path": None, "voice_type": voice_type, "ref": chosen_ref}, "duration": 0.0}

    def set_speaker_refs(self, speaker_refs: Dict, genders: Optional[Dict[str, str]] = None):
        """Installs the references of the current video (speaker -> path or candidate list, best first).

        Drops the samples and the audition left over from the previous video.
        """
        self.speaker_refs = {
            spk: [p for p in ([refs] if isinstance(refs, str) else list(refs or [])) if p]
            for spk, refs in (speaker_refs or {}).items()
        }
        self.genders = {spk: (g or "").lower() for spk, g in (genders or {}).items()}
        self.audition = {}
        self.line_refs_off = set()
        self.last_good_samples = {}

    def apply_audition(self, scores: Dict[str, Dict[str, float]], line_ref_min: float):
        """Ranks each speaker's candidates by their audition score (speaker -> {path: score}); Spec 002, 2.3."""
        for spk, by_path in scores.items():
            refs = self.speaker_refs.get(spk) or []
            self.audition.update(by_path)
            # Stable sort: equal scores keep the Stage 4 order
            self.speaker_refs[spk] = sorted(refs, key=lambda p: -by_path.get(p, -1.0))
            best = max(by_path.values(), default=None)
            if best is not None and best < line_ref_min:
                self.line_refs_off.add(spk)

    def _ref_order(self, spk: str) -> List[Tuple[str, str]]:
        """Every reference this speaker may use after the line clip, as (path or LAST_GOOD, voice type), in order."""
        own = self.speaker_refs.get(spk) or []
        good = [p for p in own if self.audition.get(p, 1.0) >= TAKE_MIN_SIMILARITY]
        weak = [p for p in own if p not in good]

        def best_first(paths):
            return sorted(paths, key=lambda p: -self.audition.get(p, 0.5))

        gender = self.genders.get(spk, "")
        others = [(o, refs[0]) for o, refs in self.speaker_refs.items() if o != spk and refs]
        same = [p for o, p in others if gender and self.genders.get(o) == gender]
        rest = [p for o, p in others if p not in same]
        rest += [p for o, refs in self.speaker_refs.items() if o != spk for p in refs[1:]]

        order = [(p, "GOLDEN") for p in good] + [("LAST_GOOD", "LAST_GOOD")]
        order += [(p, "BORROWED") for p in best_first(same)]
        order += [(p, "GOLDEN") for p in weak]
        order += [(p, "FALLBACK_GOLDEN") for p in best_first(rest)]
        return order

    def _select_reference(self, item: Dict, vocals_path: str, lang: str = "", skip=()):
        """Picks the voice reference for a line (Spec 002, 2.3); a retry skips the references already tried.

        OmniVoice clones well from 3-10 s, so a clean clip of the line itself is tried first: it carries the emotion
        of the scene. Then the speaker's own references, best audition score first; then the voice of another speaker
        of the same gender, which is better than a broken clone; anything else comes last.
        """
        spk = item["speaker"]
        idx = item["index"]
        dur = item["end"] - item["start"]
        skip = set(skip or ())

        # 1. Dynamic extraction from the line itself
        use_line = LINE_REF_RANGE[0] <= dur <= LINE_REF_RANGE[1] and "DYNAMIC" not in skip and spk not in self.line_refs_off
        if use_line:
            dyn_ref = self._extract_line_ref(item, vocals_path, lang)
            if dyn_ref:
                return dyn_ref, "DYNAMIC"

        for path, voice_type in self._ref_order(spk):
            if path == "LAST_GOOD":
                path = self.last_good_samples.get(spk)
                if spk in self.line_refs_off:
                    continue
            if not path or path in skip or not os.path.exists(path):
                continue
            if voice_type == "BORROWED":
                logging.info(f"TTS: [ID: {idx}] Borrowing a same-gender voice for speaker {spk}: {os.path.basename(path)}")
            elif voice_type == "FALLBACK_GOLDEN":
                logging.info(f"TTS: [ID: {idx}] Using Global Golden fallback for speaker {spk}")
            return path, voice_type

        all_last_goods = [p for p in self.last_good_samples.values() if os.path.exists(p) and p not in skip]
        if all_last_goods:
            logging.info(f"TTS: [ID: {idx}] Using Global LastGood fallback for speaker {spk}")
            return all_last_goods[0], "FALLBACK_LASTGOOD"

        return None, "UNKNOWN"

    def _extract_line_ref(self, item: Dict, vocals_path: str, lang: str = "") -> Optional[str]:
        """Cuts the line from the vocals track; returns its path if it is clean enough to clone from."""
        spk = item["speaker"]
        dyn_path = os.path.join(self.temp_dir, f"dyn_{lang}_{item['index']}.wav")
        audio_processor.extract_clean_segment(vocals_path, item["start"], item["end"], dyn_path)
        zcr = measure_zcr(dyn_path)

        clean = os.path.exists(dyn_path) and os.path.getsize(dyn_path) > 4000 and zcr < 0.25
        if clean:
            # A line shouted over a cheering crowd would be cloned with the crowd (Spec 002, 2.3)
            clarity = audio_processor.clarity_db(vocals_path, item["start"], item["end"])
            if clarity is not None and clarity < REF_MIN_CLARITY_DB:
                logging.info(f"TTS: [ID: {item['index']}] Line clip too noisy as a reference ({clarity:.1f} dB)")
                clean = False
        if clean:
            write_ref_text(dyn_path, item.get("source_text", ""))
            # Update Last Good Sample
            last_good_path = os.path.join(self.temp_dir, f"last_good_{lang}_{spk}.wav")
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
