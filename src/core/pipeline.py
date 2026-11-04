import os
import json
import re
import logging
import time
import humanfriendly
import shutil
import subprocess
from typing import Dict, List, Optional

from config import LANG_MAP, VIDEO_FOLDER, TEMP_DIR, SLACK_AFTER_MAX, SLACK_BEFORE_MAX, SLACK_GUARD
from config import METADATA_LOOKUP, METADATA_TIMEOUT, TMDB_API_KEY, MOCK_MODE
from config import MERGE_MAX_GAP, MERGE_MAX_DURATION, MIN_SEGMENT_DURATION, SENTENCE_MAX_GAP, SENTENCE_MAX_DURATION
from config import (
    TRANSLATION_REVIEW,
    TAKE_CHECK,
    TAKE_VERSIONS,
    TAKE_VERSIONS_SHORT,
    TAKE_GOOD_SCORE,
    REF_MIN_CLARITY_DB,
    REF_AUDITION_CANDIDATES,
    AUDITION_LINE_REF_MIN,
)
from config import VOICE_DESIGN, GLOSSARY_MODE, JUDGE_MIN_TEXT, ASR_VERIFY, EMOTION_REFS
from utils import count_syllables
from infrastructure.ffmpeg import FFmpegWrapper
from infrastructure.tts_client import write_ref_text
from infrastructure.monitor import ResourceMonitor
from core.audio import prep_audio, analyze_audio, mix_audio, clarity_db, voice_level, calm_bonus
from core.report import report_rows, summary as report_summary, write_report
from utils import clean_srt, measure_zcr, run_cmd
from core.subtitles import (
    align,
    find_sidecar,
    grammar_gender,
    label_hints,
    lang_codes,
    missing_speech,
    parse_srt,
    read_text,
    speaker_labels,
    timing_matches,
)
from core import audio as audio_processor
from core.synchronizer import SegmentSynchronizer
from core.voice_audition import audition_voices, design_voices, pitch_gender
from core.gpu_services import LLMService, TTSService
from core.media_context import lookup_media_info, format_media_context, format_short_context, episode_names
from core.series_memory import CLIP_BEST_SECONDS, SeriesMemory, clip_quality, merge_matched_speakers
from core.take_judge import TakeJudge
from core import script_check
from core.llm_engine import apply_editor_flags
from core.voice_audition import median_pitch
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# A short reference clip grows with the same speaker's lines across pauses up to this long (Spec 001, 2.4)
CLIP_JOIN_GAP = 0.6

ISO_639_2 = {
    "pl": "pol",
    "en": "eng",
    "de": "ger",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "ru": "rus",
    "ja": "jpn",
    "zh": "chi",
    "ko": "kor",
}


# Voice reference candidates per speaker that are cut and checked for noise (ZCR), best scored first
REF_CANDIDATES = 8


def ref_file_name(spk: str, cls: str, n: int) -> str:
    """ref_SPEAKER_03.wav, ref_SPEAKER_03_2.wav (calm); ref_SPEAKER_03_shout.wav, ref_SPEAKER_03_shout_2.wav, ..."""
    tag = "" if cls == "calm" else f"_{cls}"
    return f"ref_{spk}{tag}.wav" if n == 1 else f"ref_{spk}{tag}_{n}.wav"


def ref_file_energy(path: str) -> Optional[str]:
    """The energy class in a Stage 4 clip's name (kept in the name, so a cached Stage 4 still knows it)."""
    name = os.path.basename(path or "")
    if not name.startswith("ref_"):
        return None
    for cls in ("lively", "shout"):
        if re.search(rf"_{cls}(_\d+)?\.wav$", name):
            return cls
    return "calm"


# A capitalised run of words (names of people, robots, teams), as written in the script or the subtitles
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:[ -](?:[A-Z][a-z]+|[A-Z]{2,}))+)\b")


def script_names(script: List[Dict], subtitles: str = "") -> List[str]:
    """Multi-word capitalised names of the script and the reference subtitles, for the glossary (Spec 001, 2.3).

    The LLM only noticed some of them: S01E03 had no glossary entry for "Chuck Pitzer".
    """
    text = " ".join(s.get("text_en", "") for s in script) + " " + clean_srt(subtitles or "")
    counts: Dict[str, int] = {}
    for match in NAME_RE.finditer(text):
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return [n for n, _ in sorted(counts.items(), key=lambda x: -x[1])][:60]


class DubbingPipeline:
    def __init__(
        self,
        llm_manager,
        tts_manager,
        diar_manager,
        whisper_manager,
        target_langs: List[str],
        db=None,
        output_folder: str = VIDEO_FOLDER,
        temp_dir: str = TEMP_DIR,
        debug_mode: bool = False,
    ):
        self.llm_manager = llm_manager
        self.tts_manager = tts_manager
        self.diar_manager = diar_manager
        self.whisper_manager = whisper_manager
        self.target_langs = target_langs
        self.db = db
        self.output_folder = output_folder
        self.temp_dir = temp_dir
        self.base_temp_dir = temp_dir
        self.debug_mode = debug_mode

        self.durations = {}
        self.speaker_info = {}
        self.speaker_refs = {}
        self.global_context = {}
        self.series = None  # SeriesMemory of the current episode (Spec 006)
        self.voiceprints: Dict[str, List[float]] = {}  # speaker -> diarization centroid
        self.shout_prints: Dict[str, List[float]] = {}  # speaker -> centroid of its merged shouting cluster
        self.ref_energy: Dict[str, str] = {}  # reference clip -> energy class (Spec 001, 2.4)
        self.voice_matches: Dict[str, str] = {}  # speaker -> person of the voice bank
        self.available_pans = [-0.10, 0.10, -0.03, 0.03, -0.17, 0.17]
        self.speaker_pans = {}
        self.abort_event = llm_manager.abort_event
        self.monitor = None
        # The take judge decides on every take and in the voice audition (Spec 002, 2.2 step 4); TAKE_CHECK=off keeps
        # only its signal checks. Whisper only reports its similarity next to it (TAKE_CHECK=whisper, measurement).
        self.take_check = TAKE_CHECK
        self.judge = TakeJudge(temp_dir, signal_only=TAKE_CHECK == "off")
        self.take_checker = getattr(whisper_manager, "transcribe_take", None) if TAKE_CHECK == "whisper" else None
        self.synchronizer = SegmentSynchronizer(
            llm_manager=llm_manager,
            tts_manager=tts_manager,
            temp_dir=temp_dir,
            debug_mode=debug_mode,
            take_checker=self.take_checker,
            judge=self.judge,
            versions=TAKE_VERSIONS,
            short_versions=TAKE_VERSIONS_SHORT,
            good_score=TAKE_GOOD_SCORE,
        )

    def _enter_workspace(self, task_id):
        """Gives the task its own folder under TEMP_DIR (``task_<id>``, or ``standalone`` without a task).

        File names in the workspace (orig.wav, vocals.mp3, ...) are fixed, and cached step results point at them:
        in a shared folder a retried task could pick up another task's audio.
        """
        name = f"task_{task_id}" if task_id is not None else "standalone"
        self.temp_dir = os.path.join(self.base_temp_dir, name)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.synchronizer.temp_dir = self.temp_dir
        self.judge.temp_dir = self.temp_dir
        self.tts_manager.temp_dir = self.temp_dir
        logger.info(f"Workspace: {self.temp_dir}")

    def _leave_workspace(self):
        """Deletes the workspace after a finished task; a failed one keeps it, so a retry can reuse cached steps."""
        if self.debug_mode or self.temp_dir == self.base_temp_dir:
            return
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _cleanup_debug(self, fname):
        clean_name = os.path.basename(fname)
        d = os.path.join(self.output_folder, "debug_" + os.path.splitext(clean_name)[0])
        if os.path.exists(d):
            shutil.rmtree(d)
        if self.debug_mode:
            os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _dominant_speaker(diar, start, end):
        """Speaker with the largest total overlap with [start, end], or 'unknown'."""
        overlap = {}
        for d in diar:
            o = min(end, d["end"]) - max(start, d["start"])
            if o > 0:
                overlap[d["speaker"]] = overlap.get(d["speaker"], 0.0) + o
        return max(overlap, key=overlap.get) if overlap else "unknown"

    @staticmethod
    def _merge_short_lines(lines):
        """Joins lines shorter than MIN_SEGMENT_DURATION with a same-speaker neighbour.

        A 0.3-0.8 s slot cannot hold a translated sentence, so the LLM trims it to a fragment ("O!") or the TTS
        squeezes it into a clipped word. Speaking it together with the adjacent line fixes both.
        """
        out = []
        for line in lines:
            line = dict(line)
            if out:
                prev = out[-1]
                if (
                    line["end"] - line["start"] < MIN_SEGMENT_DURATION
                    and prev["speaker"] == line["speaker"]
                    and line["end"] - prev["start"] <= MERGE_MAX_DURATION
                    and len(prev["text"]) + len(line["text"]) < 400
                ):
                    prev["text"] = prev["text"].strip() + " " + line["text"].strip()
                    prev["end"] = line["end"]
                    prev["parts"] = (prev.get("parts") or []) + (line.get("parts") or [])
                    continue
            out.append(line)

        # A short first line has nobody before it: join it forward instead
        if len(out) > 1 and out[0]["end"] - out[0]["start"] < MIN_SEGMENT_DURATION:
            first, nxt = out[0], out[1]
            if first["speaker"] == nxt["speaker"] and nxt["end"] - first["start"] <= MERGE_MAX_DURATION:
                nxt["text"] = first["text"].strip() + " " + nxt["text"].strip()
                nxt["start"] = first["start"]
                nxt["parts"] = (first.get("parts") or []) + (nxt.get("parts") or [])
                out.pop(0)
        return out

    @staticmethod
    def _clamp_stretched(segment):
        """Shortens a segment whose length is far beyond what its words can take.

        Whisper sometimes stretches a short line over music ("Ready for battle" over 25 s); the dub would then be
        timed against a slot nobody speaks in.
        """
        words = count_syllables(segment.get("text", ""), "en")
        cap = max(2.0, 1.0 + words / 3.0)
        if segment["end"] - segment["start"] > cap:
            segment = dict(segment, end=segment["start"] + cap)
        return segment

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return bool(re.search(r"[.!?…][\"')\]]*\s*$", text.strip()))

    def _joins(self, curr, nxt):
        """Same speaker, and either a short pause or a sentence that continues (Spec 001, 2.2a)."""
        if nxt["speaker"] != curr["speaker"] or len(curr["text"]) + len(nxt["text"]) >= 600:
            return False
        gap = nxt["start"] - curr["end"]
        merged_duration = nxt["end"] - curr["start"]
        if gap < 0:
            return False
        if gap < MERGE_MAX_GAP and merged_duration <= MERGE_MAX_DURATION:
            return True
        return not self._ends_sentence(curr["text"]) and gap < SENTENCE_MAX_GAP and merged_duration <= SENTENCE_MAX_DURATION

    @staticmethod
    def _parts(seg) -> List[Dict]:
        """What a script line remembers of its Whisper segments for the script check (Spec 001, 2.2c)."""
        return [
            {
                "text": seg.get("text", ""),
                "hearings": seg.get("hearings") or {},
                "unsure": bool(seg.get("unsure")),
                "moved_from": seg.get("speaker_moved_from"),
            }
        ]

    @staticmethod
    def _fit_tiny_slots(lines):
        """Lengthens a unit far shorter than its words take into the silence around it (Spec 001, 2.2a step 4)."""
        for i, s in enumerate(lines):
            natural = count_syllables(s.get("text", ""), "en") / 5.0
            if s["end"] - s["start"] >= natural / 2:
                continue
            nxt = lines[i + 1]["start"] - SLACK_GUARD if i + 1 < len(lines) else s["start"] + natural
            prv = lines[i - 1]["end"] + SLACK_GUARD if i > 0 else 0.0
            end = max(s["end"], min(s["start"] + natural, nxt))
            start = min(s["start"], max(end - natural, prv))
            if (start, end) != (s["start"], s["end"]):
                logger.info(
                    f"Tiny slot: [{s['start']:.2f}-{s['end']:.2f}] -> [{start:.2f}-{end:.2f}] for '{s.get('text', '')[:40]}'"
                )
                s["start"], s["end"] = start, end
        return lines

    def _create_script(self, diar, trans):
        raw_script = []
        for t in trans:
            t = self._clamp_stretched(t)
            # The script check (Spec 001, 2.2c) may have moved the segment to the speaker its voice is
            spk = t.get("speaker") or self._dominant_speaker(diar, t["start"], t["end"])
            raw_script.append({**t, "speaker": spk, "parts": self._parts(t)})

        if not raw_script:
            return []

        merged = []
        curr = raw_script[0]
        for i in range(1, len(raw_script)):
            nxt = raw_script[i]
            if self._joins(curr, nxt):
                curr["text"] = curr["text"].strip() + " " + nxt["text"].strip()
                curr["end"] = nxt["end"]
                curr["parts"] = curr["parts"] + nxt["parts"]
                if "avg_logprob" in curr and "avg_logprob" in nxt:
                    curr["avg_logprob"] = (curr["avg_logprob"] + nxt["avg_logprob"]) / 2
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)
        merged = self._fit_tiny_slots(self._merge_short_lines(merged))

        script = []
        for i, s in enumerate(merged):
            # Silence until the next speech (any speaker) can absorb a slightly longer translation
            gap_after = merged[i + 1]["start"] - s["end"] if i + 1 < len(merged) else SLACK_AFTER_MAX + SLACK_GUARD
            slack_after = min(max(gap_after - SLACK_GUARD, 0.0), SLACK_AFTER_MAX)
            # A long take may also start a little early, in the silence before the line
            gap_before = s["start"] - merged[i - 1]["end"] if i > 0 else s["start"] + SLACK_GUARD
            slack_before = min(max(gap_before - SLACK_GUARD, 0.0), SLACK_BEFORE_MAX)
            parts = s.get("parts") or []
            line = {k: v for k, v in s.items() if k not in ("parts", "hearings", "unsure", "speaker_moved_from")}
            line.update(
                {
                    "speaker": s["speaker"],
                    "text_en": s["text"].strip(),
                    "avg_logprob": s.get("avg_logprob", 0),
                    "index": i,
                    "max_end": s["end"] + slack_after,
                    "min_start": s["start"] - slack_before,
                }
            )
            if any(p.get("unsure") for p in parts):
                line["asr_unsure"] = True
                hearings = script_check.line_hearings(parts)
                if hearings:
                    line["asr_hearings"] = hearings
            moved = sorted({p["moved_from"] for p in parts if p.get("moved_from")})
            if moved:
                line["speaker_moved"] = ",".join(moved)
            script.append(line)
        return script

    def _verify_transcript(self, trans, diar, original_path, vocals_path, subtitles, names):
        """Stage 2c (Spec 001, 2.2c): each segment's speaker by its voiceprint, unsure segments heard again."""
        segments = [dict(t, speaker=t.get("speaker") or self._dominant_speaker(diar, t["start"], t["end"])) for t in trans]
        if MOCK_MODE:
            return segments
        vocals = script_check.load_samples(vocals_path, self.temp_dir)
        segments = script_check.recheck_speakers(segments, self.voiceprints, self.judge.voiceprint_samples, vocals)
        if not any(script_check.is_unsure(t) for t in segments):
            return segments
        original = script_check.load_samples(original_path, self.temp_dir)
        self.whisper_manager.load_model()
        try:
            segments = script_check.second_hearing(
                segments,
                lambda samples, prompt: self.whisper_manager.transcribe_samples(samples, prompt),
                original,
                vocals,
                names,
                parse_srt(subtitles) if subtitles else [],
            )
        finally:
            self.whisper_manager.shutdown()
        return segments

    @staticmethod
    def _drop_hallucinations(script):
        """Lines the corrector marked as Whisper hallucinations are not dubbed (Spec 001, 2.2c step 4)."""
        kept = [s for s in script if s.get("asr") != "hallucination"]
        if len(kept) == len(script):
            return script
        for s in script:
            if s.get("asr") == "hallucination":
                logger.info(f"ASR check: [{s['start']:.1f}s] '{s['text_en']}' is a Whisper hallucination, not dubbed")
        for i, s in enumerate(kept):
            s["index"] = i
        return kept

    def _settle_energy(self, script, vocals_path):
        """Merges a shouting cluster into the same person's calm one (Spec 002, 2.3), then gives every line its
        energy class (Spec 001, 2.4)."""
        if not EMOTION_REFS or MOCK_MODE or not script:
            return
        samples = script_check.load_samples(vocals_path, self.temp_dir)
        if samples is None:
            return
        measures = script_check.measure_lines(script, samples, median_pitch)
        measured = {
            s: (i or {}).get("gender") for s, i in (self.speaker_info or {}).items() if (i or {}).get("gender_measured")
        }
        measured.update(self._grammar_genders(script))
        merges = script_check.shout_merges(script, measures, self.speaker_info or {}, self.voiceprints or {}, measured)
        for loud, calm in merges.items():
            for s in script:
                if s["speaker"] == loud:
                    s["speaker"] = calm
                    s["speaker_moved"] = ",".join(x for x in (s.get("speaker_moved"), loud) if x)
            if self.voiceprints.get(loud):
                self.shout_prints[calm] = self.voiceprints.pop(loud)
            if loud in self.voice_matches:
                self.voice_matches.setdefault(calm, self.voice_matches[loud])
                del self.voice_matches[loud]
            (self.speaker_info or {}).pop(loud, None)
        counts = script_check.assign_energy(script, measures)
        for spk, by_class in sorted(counts.items()):
            logger.info(f"Energy {spk}: " + ", ".join(f"{n} {c}" for c, n in sorted(by_class.items())))

    def _find_external_subtitles(self, vpath) -> Optional[str]:
        """Picks a sidecar .srt in the source language: `<video>.srt`, `<video>.en.srt`, `<video>.en.hi.srt`, ...

        Subtitles in a target language and forced (partial) subtitles are skipped; English and full ones are preferred.
        """
        return find_sidecar(vpath, exclude=self.target_langs)

    def _embedded_subtitles(self, vpath, langs=None, exclude=()) -> str:
        """The first embedded subtitle stream tagged with one of ``langs`` (None: any not tagged ``exclude``), as SRT."""
        try:
            meta = FFmpegWrapper.get_metadata(vpath)
        except Exception as e:
            logger.warning(f"Failed to read the subtitle streams: {e}")
            return ""
        wanted = set().union(*(lang_codes(x) for x in langs)) if langs else None
        excluded = set().union(*(lang_codes(x) for x in exclude)) if exclude else set()
        subs = [st for st in meta.get("streams", []) if st.get("codec_type") == "subtitle"]
        for n, stream in enumerate(subs):
            tag = str((stream.get("tags") or {}).get("language", "")).lower()
            if (wanted is not None and tag not in wanted) or (wanted is None and tag in excluded):
                continue
            if (stream.get("disposition") or {}).get("forced"):
                continue
            out = os.path.join(self.temp_dir, f"extracted_{tag or 'und'}_{n}.srt")
            try:
                run_cmd(["ffmpeg", "-i", vpath, "-map", f"0:s:{n}", out, "-y"], "extract subtitles")
                if os.path.exists(out):
                    return read_text(out)
            except Exception as e:
                logger.warning(f"Failed to extract embedded subtitles: {e}")
        return ""

    def _source_subtitles(self, vpath) -> str:
        """The reference subtitles (Spec 001, 2.3) as SRT text: a sidecar in a source language, else embedded."""
        srt_path = self._find_external_subtitles(vpath)
        content = ""
        if srt_path:
            logger.info(f"Found external subtitles: {srt_path}")
            try:
                content = read_text(srt_path)
            except Exception as e:
                logger.warning(f"Failed to read external SRT: {e}")
        if not content:
            content = self._embedded_subtitles(vpath, exclude=self.target_langs)
            if content:
                logger.info("Found embedded subtitles.")
        return content

    def _extract_subtitles(self, vpath, content: Optional[str] = None):
        """The reference subtitles as plain text for the analysis and the transcript corrector."""
        content = self._source_subtitles(vpath) if content is None else content
        if content:
            cleaned = clean_srt(content)
            logger.info(f"Loaded reference subtitles ({len(cleaned)} chars)")
            return cleaned
        logger.info("No reference subtitles found.")
        return ""

    def _target_subtitles(self, vpath, lang) -> List:
        """Cues of the subtitles in a target language (Spec 001, 2.3): `<video>.pl.srt`, else an embedded stream."""
        path = find_sidecar(vpath, langs=[lang])
        content = ""
        if path:
            try:
                content = read_text(path)
                logger.info(f"Found {lang} subtitles: {path}")
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
        if not content:
            content = self._embedded_subtitles(vpath, langs=[lang])
            if content:
                logger.info(f"Found embedded {lang} subtitles.")
        return parse_srt(content)

    def _add_missing_speech(self, trans, content: str):
        """Speech of the reference subtitles that Whisper did not hear joins the transcription (Spec 001, 2.3)."""
        cues = parse_srt(content)
        if not cues or not trans:
            return trans
        if not timing_matches(trans, cues):
            logger.info("Reference subtitles are not timed to this audio: not used to find missing speech")
            return trans
        missing = missing_speech(trans, cues)
        for seg in missing:
            logger.info(f"Subtitles: speech Whisper missed at {seg['start']:.1f}-{seg['end']:.1f}s: '{seg['text'][:60]}'")
        return sorted(list(trans) + missing, key=lambda t: t["start"]) if missing else trans

    def _speaker_label_hints(self, vpath, script) -> str:
        """SDH speaker labels ("KIRK:") of the speakers, for the analysis (Spec 001, 2.3)."""
        path = find_sidecar(vpath, exclude=self.target_langs, sdh=True)
        if not path:
            return ""
        try:
            hints = label_hints(speaker_labels(script, parse_srt(read_text(path))))
        except Exception as e:
            logger.warning(f"SDH speaker labels failed: {e}")
            return ""
        if hints:
            logger.info(hints)
        return hints

    def _attach_target_subtitles(self, vpath, script):
        """Each script line gets the target-language cues spoken during it: script[i]["subtitles"][lang]."""
        for lang in self.target_langs:
            cues = self._target_subtitles(vpath, lang)
            if not cues:
                continue
            aligned = align(script, cues)
            for s in script:
                if s["index"] in aligned:
                    s.setdefault("subtitles", {})[lang] = aligned[s["index"]]
            logger.info(f"Subtitles ({lang}): {len(cues)} cues on {len(aligned)} of {len(script)} lines")

    @staticmethod
    def _lookup_media_info(vpath):
        if METADATA_LOOKUP == "off":
            return {}
        online = METADATA_LOOKUP == "on" and not MOCK_MODE
        return lookup_media_info(vpath, online=online, tmdb_api_key=TMDB_API_KEY, timeout=METADATA_TIMEOUT)

    @staticmethod
    def _clarity_bonus(vocals_path, segment, spk) -> int:
        """Score for how clearly the voice stands out from the rest of the mix (Spec 001, 2.4)."""
        clarity = clarity_db(vocals_path, segment["start"], segment["end"])
        if clarity is None:
            return 0
        if clarity < REF_MIN_CLARITY_DB:
            logger.debug(f"Speaker {spk}: {segment['start']:.1f}s is noisy as a reference ({clarity:.1f} dB)")
            return -40
        return 15 if clarity >= 12 else 5 if clarity >= 6 else 0

    @staticmethod
    def _apply_calm_scores(scored, vocals_path, spk):
        """Shouted lines lose points against the speaker's own median level over the whole episode (Spec 001, 2.4)."""
        levels = [voice_level(vocals_path, s["start"], s["end"]) for s, _ in scored]
        known = sorted(lv for lv in levels if lv is not None)
        if len(known) < 3:  # too few lines to know the speaker's usual level
            return scored
        median = known[len(known) // 2] if len(known) % 2 else (known[len(known) // 2 - 1] + known[len(known) // 2]) / 2
        out = []
        for (s, total), level in zip(scored, levels):
            bonus = calm_bonus(level, median)
            if bonus < 0:
                logger.debug(f"Speaker {spk}: {s['start']:.1f}s is shouted ({level - median:+.1f} dB), {bonus} points")
            out.append((s, total + bonus))
        return out

    @staticmethod
    def _energy_groups(scored):
        """[(energy class, scored candidates)]: one group per class with EMOTION_REFS (Spec 001, 2.4 step 3)."""
        if not EMOTION_REFS:
            return [("calm", scored)]
        groups: Dict[str, list] = {}
        for seg, score in scored:
            groups.setdefault(seg.get("energy") or "calm", []).append((seg, score))
        return [(cls, groups[cls]) for cls in script_check.ENERGY_CLASSES if cls in groups]

    def _pick_clean_clips(self, spk, top, vocals_path, count=1, allow_noisy=True):
        """Cuts the best scored candidates and returns up to ``count`` that pass the ZCR check as [(segment, score, wav)].

        When none passes, the least noisy one is returned: the speaker's own voice beats another speaker's reference.
        """
        picked = []
        least_noisy = None  # (zcr, cand, score, wav)
        for cand, score in top:
            tmp_wav = os.path.join(self.temp_dir, f"test_{spk}_{cand['start']}.wav")
            filt = "highpass=f=100,afftdn=nf=-20,speechnorm=e=10:r=0.0001:l=1"
            cmd = ["ffmpeg", "-i", vocals_path, "-ss", str(cand["start"]), "-t", str(cand["end"] - cand["start"])]
            run_cmd(cmd + ["-af", filt, "-ac", "1", "-ar", "24000", tmp_wav, "-y"], "zcr check")

            zcr = measure_zcr(tmp_wav)
            if zcr <= 0.15:
                picked.append((cand, score, tmp_wav))
                if len(picked) >= count:
                    break
                continue
            if not picked and (least_noisy is None or zcr < least_noisy[0]):
                if least_noisy and os.path.exists(least_noisy[3]):
                    os.remove(least_noisy[3])
                least_noisy = (zcr, cand, score, tmp_wav)
            elif os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        if least_noisy and os.path.exists(least_noisy[3]):
            if picked or not allow_noisy:
                os.remove(least_noisy[3])
            else:
                logger.warning(
                    f"Speaker {spk}: no clip passed the ZCR check, using the least noisy one (ZCR {least_noisy[0]:.2f})."
                )
                return [least_noisy[1:]]
        return picked

    @staticmethod
    def _extend_short_clip(seg, script, vocals_path):
        """A candidate under 6 s grows with the same speaker's adjacent lines, up to 10 s (Spec 001, 2.4).

        OmniVoice clones best from 6-10 s. Only lines right next to it (a pause of at most CLIP_JOIN_GAP) of the same
        speaker that are clean enough as references join, so the clip stays one calm stretch of one voice.
        """
        lo, hi = CLIP_BEST_SECONDS
        if seg["end"] - seg["start"] >= lo or seg not in script:
            return seg
        i = j = script.index(seg)

        def usable(other, start, end):
            if other["speaker"] != seg["speaker"] or end - start > hi:
                return False
            clarity = clarity_db(vocals_path, other["start"], other["end"])
            return clarity is None or clarity >= REF_MIN_CLARITY_DB

        start, end = seg["start"], seg["end"]
        while end - start < lo:
            nxt = script[j + 1] if j + 1 < len(script) else None
            prv = script[i - 1] if i > 0 else None
            if nxt and nxt["start"] - end <= CLIP_JOIN_GAP and usable(nxt, start, nxt["end"]):
                j, end = j + 1, nxt["end"]
            elif prv and start - prv["end"] <= CLIP_JOIN_GAP and usable(prv, prv["start"], end):
                i, start = i - 1, prv["start"]
            else:
                break
        if (i, j) == (script.index(seg),) * 2:
            return seg
        text = " ".join(script[k]["text_en"].strip() for k in range(i, j + 1))
        return dict(seg, start=start, end=end, text_en=text)

    def _extract_refs(self, script, vocals_path, ddir):
        logger.info("Extracting Smart Weighted Normalized voice references (v2 - Scored 0-100)")
        self.speaker_refs = {}
        self.ref_quality = {}
        ref_debug = os.path.join(ddir, "refs") if self.debug_mode else None
        if ref_debug:
            os.makedirs(ref_debug, exist_ok=True)

        bad_markers = ["[", "]", "(", ")", "music", "laughter", "scream", "explosion", "sound", "noise", "intro", "theme"]
        total_dur = script[-1]["end"] if script else 0

        best_per_speaker = {}
        all_speakers = set(s["speaker"] for s in script if s["speaker"] != "unknown")

        for spk in all_speakers:
            candidates = [s for s in script if s["speaker"] == spk]
            scored_candidates = []

            for s in candidates:
                txt = s["text_en"].lower()
                if any(x in txt for x in bad_markers):
                    continue
                dur = s["end"] - s["start"]
                if dur < 2.0 or dur > 20.0:
                    continue

                clarity_score = 50
                conf = s.get("avg_logprob", -0.5)
                if conf < -0.4:
                    clarity_score -= 10
                if conf < -0.7:
                    clarity_score -= 15

                nsp = s.get("no_speech_prob", 0)
                if nsp > 0.1:
                    clarity_score -= 10
                if nsp > 0.3:
                    clarity_score -= 20

                cr = s.get("compression_ratio", 1.2)
                if cr > 1.8:
                    clarity_score = 0
                if cr < 0.6:
                    clarity_score -= 10

                words = len(s["text_en"].split())
                wps = words / dur
                if wps < 0.5 or wps > 4.5:
                    clarity_score -= 15

                if clarity_score < 0:
                    clarity_score = 0

                dur_score = 0
                if 6.0 <= dur <= 14.0:
                    dur_score = 40
                elif 4.0 <= dur < 6.0:
                    dur_score = 25
                elif 14.0 < dur <= 18.0:
                    dur_score = 20
                else:
                    dur_score = 5

                pos_score = 0
                if total_dur > 0:
                    rel_pos = s["start"] / total_dur
                    if 0.15 < rel_pos < 0.85:
                        pos_score = 10
                    elif 0.05 < rel_pos < 0.95:
                        pos_score = 5

                total = clarity_score + dur_score + pos_score + self._clarity_bonus(vocals_path, s, spk)
                scored_candidates.append((s, total))

            if scored_candidates:
                clips = []
                for cls, group in self._energy_groups(scored_candidates):
                    if cls == "calm":
                        group = self._apply_calm_scores(group, vocals_path, spk)
                    top = sorted(group, key=lambda x: x[1], reverse=True)[:REF_CANDIDATES]
                    top = [(self._extend_short_clip(seg, script, vocals_path), score) for seg, score in top]
                    # Only the calm class falls back to a noisy clip: an empty shout class is better than a noisy one
                    picked = self._pick_clean_clips(spk, top, vocals_path, REF_AUDITION_CANDIDATES, allow_noisy=cls == "calm")
                    clips += [(seg, score, wav, cls) for seg, score, wav in picked]
                if not clips and EMOTION_REFS:
                    # No clean clip in any class: the least noisy one of all, like before the classes
                    top = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:REF_CANDIDATES]
                    top = [(self._extend_short_clip(seg, script, vocals_path), score) for seg, score in top]
                    picked = self._pick_clean_clips(spk, top, vocals_path, 1)
                    clips = [(seg, score, wav, seg.get("energy") or "calm") for seg, score, wav in picked]

                if clips:
                    paths = []
                    seen: Dict[str, int] = {}
                    for seg, score, wav_path, cls in clips:
                        seen[cls] = seen.get(cls, 0) + 1
                        out = os.path.join(self.temp_dir, ref_file_name(spk, cls, seen[cls]))
                        shutil.move(wav_path, out)
                        write_ref_text(out, seg["text_en"])
                        paths.append(out)
                        clarity = clarity_db(vocals_path, seg["start"], seg["end"])
                        self.ref_quality[out] = clip_quality(seg["end"] - seg["start"], clarity)
                        if ref_debug:
                            shutil.copy(out, os.path.join(ref_debug, os.path.basename(out)[4:]))
                    best_seg, best_score = clips[0][0], clips[0][1]
                    best_per_speaker[spk] = (best_seg, best_score)
                    self.speaker_refs[spk] = paths
                    logger.info(
                        f"Speaker {spk} Selected: Score {best_score} (Dur: {best_seg['end'] - best_seg['start']:.1f}s)"
                        + (f", {len(paths)} candidates" if len(paths) > 1 else "")
                    )
                else:
                    logger.warning(f"Speaker {spk}: No clean candidates found after ZCR check.")

        for spk in all_speakers:
            my_best = best_per_speaker.get(spk)
            if not my_best:
                logger.warning(f"  [Speaker {spk}] No candidates found. Voice cloning might fail.")
            elif my_best[1] < 65:
                logger.warning(f"  [Speaker {spk}] Low Score {my_best[1]}. Using best available candidate.")

        return self.speaker_refs

    def _apply_mastering_and_speed(self, r, f, spk, speed):
        p = self.speaker_pans.get(spk)
        if p is None:
            info = self.speaker_info.get(spk, {})
            name = info.get("name", "").lower()
            desc = info.get("desc", "").lower()
            is_narrator = any(x in name or x in desc for x in ["narrator", "lektor", "narratorka"])
            if is_narrator:
                p = 0.0
                logger.info(f"Speaker {spk} identified as Narrator. Centering audio.")
            else:
                p = self.available_pans.pop(0) if self.available_pans else 0.0
            self.speaker_pans[spk] = p
        info = self.speaker_info.get(spk, {})
        is_narrator = any(
            x in info.get("name", "").lower() or x in info.get("desc", "").lower()
            for x in ["narrator", "lektor", "narratorka"]
        )
        echo = "aecho=0.8:0.9:10:0.2," if not is_narrator else ""
        filt = f"highpass=f=60,{echo}speechnorm=e=4:r=0.0001:l=1,pan=stereo|c0={1.0 - max(0, p):.2f}*c0|c1={1.0 + min(0, p):.2f}*c0,atempo={speed}"
        subprocess.run(["ffmpeg", "-i", r, "-af", filt, f, "-y"], capture_output=True)

    def _run_step(self, name, func, task_id, *args):
        # Check for checkpoint if db and task_id provided
        if self.db and task_id:
            cached = self.db.get_step_result(task_id, name)
            if cached is not None:
                # Validate that files mentioned in cached result actually exist
                def validate_files(obj):
                    if isinstance(obj, str) and (obj.startswith("/") or "/" in obj) and "." in obj:
                        if os.path.exists(obj):
                            return True
                        # If it looks like a path but doesn't exist, fail validation
                        return False
                    if isinstance(obj, (list, tuple)):
                        return all(validate_files(x) for x in obj)
                    if isinstance(obj, dict):
                        return all(validate_files(v) for v in obj.values())
                    return True

                if validate_files(cached):
                    logger.info(f"STEP {name}: Using cached result.")
                    return cached
                else:
                    logger.warning(f"STEP {name}: Cache exists but referenced files are missing. Re-running step.")

        logger.info(f"STARTING STEP: {name}")
        t = time.perf_counter()
        res = func(*args)
        self.durations[name] = time.perf_counter() - t
        logger.info(f"COMPLETED STEP: {name} in {humanfriendly.format_timespan(self.durations[name])}")

        if self.db and task_id:
            self.db.save_step_result(task_id, name, "DONE", result_data=res)
        return res

    def process_video(self, vpath: str, task_id: Optional[int] = None):
        if not os.path.isabs(vpath):
            vpath = os.path.join(VIDEO_FOLDER, vpath)

        logger.info(f"=== PROCESSING FILE: {vpath} (Task: {task_id}) ===")

        # 0. Early check for existing languages
        existing_langs = audio_processor.get_audio_languages(vpath)
        iso_map = ISO_639_2

        active_langs = []
        for lang in self.target_langs:
            if lang.lower() in existing_langs or iso_map.get(lang.lower()) in existing_langs:
                logger.info(f"--- SKIPPING {lang}: Language already exists in source video ---")
            else:
                active_langs.append(lang)

        if not active_langs:
            logger.info("All target languages already exist. Nothing to do.")
            return

        self.target_langs = active_langs
        logger.info(f"Languages to produce: {', '.join(self.target_langs)}")

        start_all = time.perf_counter()
        self._enter_workspace(task_id)
        ddir = self._cleanup_debug(vpath)
        seg_dir = os.path.join(ddir, "segments") if self.debug_mode else None

        if seg_dir and os.path.exists(seg_dir):
            shutil.rmtree(seg_dir)
        if seg_dir and self.debug_mode:
            os.makedirs(seg_dir, exist_ok=True)

        self.monitor = ResourceMonitor()
        self.monitor.daemon = True
        self.monitor.start()

        a_stereo, vocals = self._run_step("Stage 1: Audio Separation", prep_audio, task_id, vpath, self.temp_dir)
        analysis_data = self._run_step(
            "Stage 2: Audio Analysis",
            analyze_audio,
            task_id,
            vocals,
            self.diar_manager,
            self.whisper_manager,
            a_stereo,
            self.temp_dir,
        )
        diar, trans, audio_durs = analysis_data[0], analysis_data[1], analysis_data[2]
        self.voiceprints = analysis_data[3] if len(analysis_data) > 3 else {}
        self.durations.update(audio_durs)

        # What is this video? (show/episode or movie, synopsis, cast) - context for the analysis and the translator
        media_info = self._run_step("Stage 2b: Media Metadata", self._lookup_media_info, task_id, vpath)
        if self.debug_mode and media_info:
            with open(os.path.join(ddir, "media_info.json"), "w") as f_dbg:
                json.dump(media_info, f_dbg, indent=2, ensure_ascii=False)

        # Series memory (Spec 006): known persons by voice (merging split speakers), settled name spellings
        self.series = SeriesMemory.for_media(media_info)
        self.voice_matches = {}
        self.shout_prints = {}
        if self.series and self.voiceprints:
            self.voice_matches = self.series.match(self.voiceprints)
            diar, self.voiceprints, self.voice_matches = merge_matched_speakers(diar, self.voiceprints, self.voice_matches)
        fixed_glossary = self.series.glossary(self._glossary_key()) if self.series else {}
        # Settled English phonemes: the TTS input in phonemes mode, what Polish respellings are checked against otherwise
        fixed_phonemes = self.series.glossary("cmu") if self.series else {}
        if GLOSSARY_MODE == "phonemes":
            fixed_glossary = {t: t for t in fixed_phonemes}

        source_subs = self._source_subtitles(vpath)
        trans = self._add_missing_speech(trans, source_subs)
        if ASR_VERIFY:
            names = episode_names(media_info) + list(fixed_glossary or {}) + list(fixed_phonemes or {})
            trans = self._run_step(
                "Stage 2c: Script Verification",
                self._verify_transcript,
                task_id,
                trans,
                diar,
                a_stereo,
                vocals,
                source_subs,
                names,
            )
        script = self._create_script(diar, trans)
        self._attach_target_subtitles(vpath, script)
        if self.debug_mode:
            with open(os.path.join(ddir, "script_initial.json"), "w") as f_dbg:
                json.dump(script, f_dbg, indent=2, ensure_ascii=False)
        if self.abort_event.is_set():
            return
        ref_subs = self._extract_subtitles(vpath, source_subs)
        media_text = format_media_context(media_info)
        hints = self.series.hints(self.voice_matches) if self.series else ""
        sdh_hints = self._speaker_label_hints(vpath, script)
        if sdh_hints:
            hints = f"{hints}\n{sdh_hints}".strip()
        if hints:
            media_text = f"{media_text}\n{hints}".strip()
        analysis_results = self._run_step(
            "Stage 3: Global Analysis",
            self.llm_manager.analyze_script,
            task_id,
            script,
            ddir,
            ref_subs,
            media_text,
            fixed_glossary,
            episode_names(media_info) + script_names(script, source_subs),
            fixed_phonemes,
        )
        self.global_context = analysis_results["context"]
        if media_info:
            self.global_context["media"] = format_short_context(media_info)
        self.speaker_info = analysis_results["speakers"]
        for spk, gender in (self.series.settled_genders(self.voice_matches) if self.series else {}).items():
            self.speaker_info.setdefault(spk, {})["gender"] = gender
        # analyze_script corrects the script in place; on a cached run only its returned fixes carry them over
        for idx_str, text in analysis_results.get("script_fixes", {}).items():
            if 0 <= int(idx_str) < len(script):
                script[int(idx_str)]["text_en"] = text
        apply_editor_flags(
            script,
            analysis_results.get("script_flags") or {},
            analysis_results.get("speaker_doubts") or {},
            analysis_results.get("script_fixes") or {},
        )
        script = self._drop_hallucinations(script)
        self._settle_energy(script, vocals)

        refs = self._run_step("Stage 4: Transcription Correction", self._extract_refs, task_id, script, vocals, ddir)
        self.speaker_refs = {spk: list(paths) for spk, paths in (refs or {}).items()}
        # A known person's clips from earlier episodes join the candidates; the voice audition ranks them all
        for spk, pid in self.voice_matches.items():
            bank = [p for p in self.series.clips(pid) if p not in self.speaker_refs.get(spk, [])]
            if bank:
                self.speaker_refs.setdefault(spk, []).extend(bank)
                logger.info(f"Series memory: {spk} gets {len(bank)} reference clips of {pid} from earlier episodes")
        self._check_genders(refs or {}, script)
        genders = {spk: (info or {}).get("gender", "") for spk, info in (self.speaker_info or {}).items()}
        self.ref_energy = {p: ref_file_energy(p) for paths in self.speaker_refs.values() for p in paths if ref_file_energy(p)}
        for pid in set(self.voice_matches.values()):
            self.ref_energy.update(self.series.clip_energies(pid))
        self.tts_manager.set_speaker_refs(self.speaker_refs, genders, self.ref_energy)
        self.tts_manager.phonemes = self.global_context.get("phonemes") or {}

        # 5. Production (Per Language / Global Task Pool)
        # Initialize GPU RPC Services
        # 1 worker for LLM to ensure absolute stability with llama-cpp-python CUDA
        llm_service = LLMService(num_workers=1)
        tts_service = TTSService()
        llm_service.start()
        tts_service.start()

        try:
            # Connect services to monitor for queue size logging
            if self.monitor:
                self.monitor.llm_service = llm_service
                self.monitor.tts_service = tts_service

            # Connect managers to services
            self.llm_manager.service = llm_service
            self.tts_manager.service = tts_service

            all_audio_tracks = self._execute_production_pool(script, vocals, ddir, a_stereo, task_id)

        finally:
            # Stop services and unload LLM from VRAM
            llm_service.stop()
            tts_service.stop()
            self.llm_manager.shutdown()
            self.llm_manager.service = None
            self.tts_manager.service = None

        if all_audio_tracks:
            ext = os.path.splitext(vpath)[1] or ".mkv"
            final_video = os.path.join(self.temp_dir, f"final_muxed{ext}")
            self._run_step("Stage 7: Muxing", FFmpegWrapper.mux_video, task_id, vpath, all_audio_tracks, final_video)

            # Atomic swap with backup
            bak_path = vpath + ".bak"
            logger.info(f"Moving original to backup: {bak_path}")
            shutil.move(vpath, bak_path)
            try:
                logger.info(f"Installing dubbed version: {vpath}")
                shutil.move(final_video, vpath)
                # If everything is fine, we could delete the backup,
                # but keeping it for a moment is safer.
                # os.remove(bak_path)
            except Exception as e:
                logger.error(f"Failed to install dubbed version! Restoring backup. Error: {e}")
                shutil.move(bak_path, vpath)
                raise e
            self._remember_series(script)

        self.tts_manager.shutdown()
        if self.monitor:
            self.monitor.stop()
        self._leave_workspace()
        avg_tps = (
            self.llm_manager.llm_stats["tokens"] / self.llm_manager.llm_stats["time"]
            if self.llm_manager.llm_stats["time"] > 0
            else 0
        )
        self._print_report(vpath, time.perf_counter() - start_all, avg_tps)

    def _prepare_terms(self, script, lang, task_id=None):
        """The series' terms and forms of address for this language, with those of the episode's own subtitles on
        top (Spec 006, 5), into global_context["terms"][lang] and ["address"][lang]."""
        stored_terms, stored_address = self.series.terms(lang) if getattr(self, "series", None) else ({}, {})
        episode = {"terms": {}, "address": {}}
        if any((s.get("subtitles") or {}).get(lang) for s in script):
            try:
                episode = (
                    self._run_step(
                        f"Stage 5a: Subtitle Terms ({lang})",
                        self.llm_manager.extract_terms,
                        task_id,
                        script,
                        lang,
                        self.speaker_info,
                    )
                    or episode
                )
            except Exception as e:
                logger.warning(f"Subtitle terms ({lang}) failed: {e}")
        if not hasattr(self, "episode_terms"):
            self.episode_terms = {}
        self.episode_terms[lang] = episode
        self.global_context.setdefault("terms", {})[lang] = {**stored_terms, **(episode.get("terms") or {})}
        self.global_context.setdefault("address", {})[lang] = {**stored_address, **(episode.get("address") or {})}
        if self.global_context["terms"][lang]:
            logger.info(f"Terms ({lang}): {len(self.global_context['terms'][lang])} ({len(stored_terms)} from the series)")

    def _remember_series(self, script):
        """Stores what this episode settled for the next episodes of the series (Spec 006). Never fails the task."""
        if not getattr(self, "series", None):
            return
        try:
            ctx = self.global_context or {}
            glossary = (ctx.get("phonemes") if GLOSSARY_MODE == "phonemes" else ctx.get("glossary")) or {}
            text = " ".join(s.get("text_en", "") for s in script).lower()
            used = [t for t in glossary if t.lower() in text]
            self.series.save_glossary(self._glossary_key(), glossary, used, replace=ctx.get("glossary_replaced") or [])
            if GLOSSARY_MODE != "phonemes" and ctx.get("name_phonemes"):
                self.series.save_glossary("cmu", ctx["name_phonemes"], used)
        except Exception as e:
            logger.warning(f"Series memory: could not save the glossary ({e})")
        try:
            text = " ".join(s.get("text_en", "") for s in script).lower()
            for lang, episode in (getattr(self, "episode_terms", None) or {}).items():
                known = (self.global_context or {}).get("terms", {}).get(lang) or {}
                used = [t for t in known if t.lower() in text]
                self.series.save_terms(lang, episode.get("terms") or {}, episode.get("address") or {}, used)
        except Exception as e:
            logger.warning(f"Series memory: could not save the terms ({e})")
        try:
            self.series.save_voices(self._bank_update(script))
        except Exception as e:
            logger.warning(f"Series memory: could not save the voice bank ({e})")

    def _glossary_key(self) -> str:
        """Respellings depend on the target languages; English phonemes do not."""
        return "cmu" if GLOSSARY_MODE == "phonemes" else SeriesMemory.lang_key(self.target_langs)

    def _bank_update(self, script):
        """What this episode tells the voice bank: every speaker with at least 3 lines (Spec 006, 4.6)."""
        lines: Dict[str, int] = {}
        for s in script:
            lines[s["speaker"]] = lines.get(s["speaker"], 0) + 1
        tts = self.tts_manager
        audition = getattr(tts, "audition", {}) or {}
        designed = getattr(tts, "designed", set()) or set()
        update = {}
        for spk, count in lines.items():
            if count < 3 or spk == "unknown":
                continue
            refs = (getattr(tts, "speaker_refs", {}) or {}).get(spk) or []
            info = (self.speaker_info or {}).get(spk) or {}
            update[spk] = {
                "person": self.voice_matches.get(spk),
                "print": self.voiceprints.get(spk),
                "shout_print": self.shout_prints.get(spk),
                "profile": {
                    "name": info.get("name"),
                    "gender": info.get("gender"),
                    "desc": info.get("desc"),
                    "gender_measured": bool(info.get("gender_measured")),
                    "gender_hz": info.get("gender_hz"),
                },
                "clips": [
                    (p, audition.get(p), (getattr(self, "ref_quality", None) or {}).get(p), self.ref_energy.get(p))
                    for p in refs
                    if p not in designed and audition.get(p) is not None
                ],
            }
        return update

    def _print_report(self, f, t, avg_tps):
        rep = ["\n" + "=" * 50, f" PROFILING REPORT: {f}", "=" * 50]
        for k, v in sorted(self.durations.items()):
            rep.append(f" - {k:35} : {humanfriendly.format_timespan(v)}")
        rep.extend(
            [
                "-" * 50,
                f" - LLM PERFORMANCE                  : {avg_tps:.2f} tokens/s",
                "-" * 50,
                f" TOTAL PROCESSING TIME: {humanfriendly.format_timespan(t)}",
                "=" * 50 + "\n",
            ]
        )
        logger.info("\n".join(rep))

    def _execute_production_pool(self, script, vocals, ddir, a_stereo, task_id):
        """Unified concurrent production for all languages."""
        # 1. Draft Translation for ALL languages (sequential prep)
        all_drafts_by_lang = {}
        for lang in self.target_langs:
            logger.info(f"--- PREPARING DRAFTS FOR LANGUAGE: {lang} ---")
            self._prepare_terms(script, lang, task_id)
            t_start_draft = time.perf_counter()
            draft_segments = self.llm_manager.generate_drafts(script, lang, self.speaker_info, self.global_context)
            if TRANSLATION_REVIEW:
                draft_segments = self.llm_manager.review_drafts(draft_segments, lang, self.speaker_info, self.global_context)
            # The sync loop shows neighbouring translations to the refinement
            for seg in draft_segments:
                script[seg["index"]].setdefault("translations", {})[lang] = seg.get("text", "")
            if self.debug_mode:
                with open(os.path.join(ddir, f"drafts_{lang}.json"), "w") as f_dbg:
                    json.dump(draft_segments, f_dbg, indent=2, ensure_ascii=False)
            self.durations[f"Stage 5a: Draft Translation ({lang})"] = time.perf_counter() - t_start_draft
            all_drafts_by_lang[lang] = draft_segments

        # 2. Flatten all segments into a single task pool
        all_sync_tasks = []
        for lang, drafts in all_drafts_by_lang.items():
            for seg in drafts:
                all_sync_tasks.append((seg, lang))

        logger.info(f"--- STARTING GLOBAL PRODUCTION POOL ({len(all_sync_tasks)} tasks) ---")
        t_start_sync_all = time.perf_counter()

        sync_results_by_lang = {lang: [] for lang in self.target_langs}
        checkpoint_path = os.path.join(self.temp_dir, f"stage5_checkpoint_{task_id}.json")

        self._load_production_checkpoint(checkpoint_path, sync_results_by_lang)

        if self.monitor:
            self.monitor.orchestrator_queue_size = len(all_sync_tasks)

        # 3. Global Concurrent Execution (Flattened)
        try:
            self._audition_voices(all_drafts_by_lang)
            self._run_concurrent_sync(all_sync_tasks, sync_results_by_lang, vocals, script, checkpoint_path)
        finally:
            if hasattr(self.whisper_manager, "shutdown_take_check"):
                self.whisper_manager.shutdown_take_check()
            self.judge.shutdown()

        # Per-line report: a summary in the log, the whole table in the debug folder (Spec 001, 2.5)
        for lang in self.target_langs:
            try:
                if self.debug_mode:
                    summary = write_report(
                        os.path.join(ddir, f"report_{lang}.tsv"), script, sync_results_by_lang[lang], self.ref_energy
                    )
                else:
                    summary = report_summary(report_rows(script, sync_results_by_lang[lang]))
                logger.info(f"Production report ({lang}): {summary}")
            except Exception as e:
                logger.warning(f"Production report ({lang}) failed: {e}")

        # Without any dubbed line the "dubbed" track would only be the ducked original, so fail the task instead
        silent_langs = [lang for lang, drafts in all_drafts_by_lang.items() if drafts and not sync_results_by_lang[lang]]
        if silent_langs:
            raise RuntimeError(f"No segment could be synthesized for: {', '.join(silent_langs)}")

        # A track where most lines are silent (typically VRAM exhaustion during synthesis) is worse than no
        # track: fail the task and keep the checkpoint, instead of muxing a broken dub over the original.
        mostly_silent = []
        for lang in self.target_langs:
            results = sync_results_by_lang[lang]
            failed = sum(1 for r in results if r.get("status") == "FAILED_FALLBACK")
            if results and failed * 2 > len(results):
                mostly_silent.append(f"{lang} ({failed}/{len(results)} lines)")
        if mostly_silent:
            raise RuntimeError(f"Too many segment synthesis failures: {', '.join(mostly_silent)}")

        # 4. Mastering & Mixing (Per Language)
        all_audio_tracks = self._finalize_all_languages(sync_results_by_lang, a_stereo, t_start_sync_all)

        if self.db and task_id:
            self.db.save_step_result(task_id, "Stage 5: Production", "DONE", result_data={"langs": self.target_langs})

        # Remove checkpoint upon success
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass

        return all_audio_tracks

    def _grammar_genders(self, script) -> Dict[str, str]:
        """Speaker -> gender told by the first-person forms of its target-language subtitles (Spec 001, 2.3)."""
        out: Dict[str, str] = {}
        if not script:
            return out
        for lang in self.target_langs:
            texts: Dict[str, List[str]] = {}
            for s in script or []:
                sub = (s.get("subtitles") or {}).get(lang)
                if sub:
                    texts.setdefault(s["speaker"], []).append(sub)
            for spk, lines in texts.items():
                gender = grammar_gender(lines, lang)
                if gender and spk not in out:
                    out[spk] = gender
        return out

    def _set_measured_gender(self, spk: str, gender: str, why: str, pitch: Optional[float] = None):
        info = self.speaker_info.setdefault(spk, {})
        guessed = info.get("gender")
        if guessed != gender:
            logger.info(f"Speaker {spk}: {why}, so {gender} (the analysis said '{guessed}')")
            # The analysis took this speaker for someone else: "Jenny Taft (male)" makes the translator use
            # feminine forms for a man
            if guessed in ("male", "female") and info.get("name"):
                logger.info(f"Speaker {spk}: dropping the name '{info['name']}', which belongs to a {guessed} person")
                info["name"] = ""
        info["gender"] = gender
        info["gender_measured"] = True
        if pitch:
            info["gender_hz"] = round(pitch)

    def _check_genders(self, refs: Dict[str, List[str]], script=None):
        """The gender told by the subtitles' grammar, else by the measured pitch, overrules the LLM's guess from names
        (Spec 001, 2.3; Spec 002, 2.3).

        The LLM names speakers from the cast list and can give the play-by-play man the name and gender of the
        sideline reporter; the designed voice and the series memory would then carry a woman's voice for him.
        """
        grammar = self._grammar_genders(script)
        for spk, gender in grammar.items():
            self._set_measured_gender(spk, gender, "the subtitles' grammar says so")
        if MOCK_MODE:
            return
        for spk, paths in refs.items():
            if spk in grammar:
                continue
            gender, pitch = pitch_gender(list(paths or [])[:REF_AUDITION_CANDIDATES])
            info = self.speaker_info.setdefault(spk, {})
            if not gender:
                if pitch:
                    logger.info(
                        f"Speaker {spk}: pitch {pitch:.0f} Hz does not tell the gender, keeping '{info.get('gender')}'"
                    )
                continue
            self._set_measured_gender(spk, gender, f"pitch {pitch:.0f} Hz", pitch)

    def _audition_voices(self, all_drafts_by_lang):
        """Ranks each speaker's reference candidates by how intelligibly they clone (Spec 002, 2.3)."""
        if MOCK_MODE or not self.target_langs:
            return
        lang = self.target_langs[0]
        t0 = time.perf_counter()
        if VOICE_DESIGN:
            try:
                design_voices(self.tts_manager, self.speaker_info, all_drafts_by_lang.get(lang) or [], lang, self.temp_dir)
            except Exception as e:
                logger.warning(f"Voice design failed: {e}")
        if self.take_check == "off":
            return
        try:
            scores = audition_voices(
                self.tts_manager,
                self._audition_score,
                all_drafts_by_lang.get(lang) or [],
                lang,
                self.temp_dir,
                self.synchronizer.is_swallowed,
                self.abort_event,
            )
        except Exception as e:
            logger.warning(f"Voice audition failed, keeping the Stage 4 order: {e}")
            return
        self.tts_manager.apply_audition(scores, AUDITION_LINE_REF_MIN)
        if self.tts_manager.line_refs_off:
            logger.info(f"Voice audition: no line clips for {', '.join(sorted(self.tts_manager.line_refs_off))}")
        self.durations["Stage 5a: Voice Audition"] = time.perf_counter() - t0

    def _audition_score(self, path, text, lang, ref):
        """The judge's score of an audition take; Whisper's similarity stands in when the judge has no text check."""
        verdict = self.judge.judge(text, path, lang, ref)
        score = verdict["score"]
        if verdict.get("text_match") is None and not verdict.get("problem"):
            check = self.synchronizer._intelligibility(text, path, lang) if count_syllables(text, lang) >= 4 else None
            if check is not None:
                score = min(score, check[0])
                if check[0] < JUDGE_MIN_TEXT:
                    verdict["problem"] = f"UNINTELLIGIBLE (Whisper similarity {check[0]:.2f}, heard '{check[1][:40]}')"
        if verdict.get("problem"):
            logger.info(f"Voice audition: {os.path.basename(ref or '')}: {verdict['problem']}")
        return score

    def _load_production_checkpoint(self, checkpoint_path, sync_results_by_lang):
        """Loads partial results from disk if available."""
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f_cp:
                    checkpoint_data = json.load(f_cp)
                    for lang, results in checkpoint_data.items():
                        if lang in sync_results_by_lang:
                            sync_results_by_lang[lang] = results
                logger.info(f"Restored {sum(len(r) for r in sync_results_by_lang.values())} segments from checkpoint.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _run_concurrent_sync(self, all_sync_tasks, sync_results_by_lang, vocals, script, checkpoint_path):
        """Executes the concurrent synchronization loop with checkpointing."""
        remaining_tasks = []
        for seg, lang in all_sync_tasks:
            idx = seg["index"]
            if any(r["index"] == idx for r in sync_results_by_lang[lang]):
                continue
            remaining_tasks.append((seg, lang))

        logger.info(f"Tasks remaining after checkpoint: {len(remaining_tasks)}/{len(all_sync_tasks)}")
        if not remaining_tasks:
            return

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(self.synchronizer.process_segment, task[0], task[1], vocals, script, self.global_context): task
                for task in remaining_tasks
            }

            completed_count = len(all_sync_tasks) - len(remaining_tasks)
            total_tasks = len(all_sync_tasks)
            pending_ids = {seg["index"] for seg, _ in remaining_tasks}

            try:
                for future in as_completed(future_to_task):
                    if self.abort_event.is_set():
                        executor.shutdown(wait=False)
                        raise RuntimeError("Processing aborted via event")

                    seg, lang = future_to_task[future]
                    idx = seg["index"]

                    try:
                        seg_result = future.result(timeout=120)
                        if seg_result:
                            # Restore metadata for mixing
                            seg_result["start"] = script[idx]["start"]
                            seg_result["end"] = script[idx]["end"]
                            seg_result["max_end"] = script[idx].get("max_end", script[idx]["end"])
                            seg_result["min_start"] = script[idx].get("min_start", script[idx]["start"])
                            seg_result["speaker"] = script[idx]["speaker"]
                            seg_result["index"] = idx
                            sync_results_by_lang[lang].append(seg_result)

                            # Update checkpoint every segment
                            try:
                                with open(checkpoint_path, "w") as f_cp:
                                    json.dump(sync_results_by_lang, f_cp, indent=2)
                            except Exception:
                                pass
                    except TimeoutError:
                        logger.error(f"Segment {idx} ({lang}) TIMED OUT in orchestrator. Skipping.")
                    except Exception as e:
                        logger.error(f"Segment {idx} ({lang}) failed completely: {e}")

                    completed_count += 1
                    if idx in pending_ids:
                        pending_ids.remove(idx)

                    if self.monitor:
                        self.monitor.orchestrator_queue_size = total_tasks - completed_count

                    if completed_count % 5 == 0:
                        logger.info(
                            f"  [Global Progress] {completed_count}/{total_tasks} segments processed. Pending IDs sample: {list(pending_ids)[:5]}..."
                        )
            except Exception as e:
                logger.error(f"Orchestrator loop error: {e}")
                raise

    @staticmethod
    def _place_take(item, dur, last_end_time):
        """Where a take starts in the mix and how much it is sped up (Spec 002, 2.4).

        A take that would run past its latest end starts earlier, into the silence before the line (down to
        ``min_start``); it never starts before the previous dubbed line has ended.
        """
        start = item["start"]
        max_end = item.get("max_end", item["end"])
        min_start = min(item.get("min_start", start), start)
        overrun = dur - (max_end - start)
        if overrun > 0:
            start -= min(overrun, start - min_start)
        if start < last_end_time and last_end_time - start > 0.05:
            start = last_end_time
        available = max(max_end - start, 0.3)
        speed_factor = 1.0
        if dur > available + 0.2:
            speed_factor = min(dur / available, 1.20)
        return start, speed_factor

    def _finalize_all_languages(self, sync_results_by_lang, a_stereo, t_start_sync_all):
        """Processes mastered segments and performs final language mixing."""
        all_audio_tracks = []
        for lang in self.target_langs:
            logger.info(f"--- FINALIZING LANGUAGE: {lang} ---")
            res = sync_results_by_lang[lang]
            res.sort(key=lambda x: x["index"])

            final_audio_segments = []
            last_end_time = 0

            for item in res:
                dur = item["duration"]
                path = item.get("audio_path")
                if not path:
                    # DROPPED: every take was gibberish, the original stays audible (Spec 002, 2.2 step 8)
                    continue
                start, speed_factor = self._place_take(item, dur, last_end_time)

                final_path = os.path.join(self.temp_dir, f"final_{lang}_{item['index']}.wav")
                audio_processor.trim_and_pad_silence(path, dur)

                self._apply_mastering_and_speed(path, final_path, item["speaker"], speed_factor)
                final_audio_segments.append((final_path, start, dur / speed_factor))
                last_end_time = start + (dur / speed_factor)

            final_a = os.path.join(self.temp_dir, f"final_{lang}.ac3")
            mix_audio(a_stereo, final_audio_segments, final_a, self.temp_dir)
            self.durations[f"Stage 5b: Sync Production ({lang})"] = (time.perf_counter() - t_start_sync_all) / len(
                self.target_langs
            )
            self.durations[f"Stage 6: Final Mix ({lang})"] = 0.1
            all_audio_tracks.append((final_a, lang, LANG_MAP.get(lang, lang)))
        return all_audio_tracks
