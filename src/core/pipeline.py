import os
import json
import logging
import time
import threading
import humanfriendly
import shutil
import subprocess
from typing import List, Optional

from config import LANG_MAP, VIDEO_FOLDER, TEMP_DIR
from infrastructure.ffmpeg import FFmpegWrapper
from infrastructure.monitor import ResourceMonitor
from core.audio import prep_audio, analyze_audio, mix_audio
from utils import clean_srt, measure_zcr, run_cmd
from core import audio as audio_processor
from core.synchronizer import SegmentSynchronizer
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class DubbingPipeline:
    def __init__(
        self,
        llm_manager,
        tts_manager,
        target_langs: List[str],
        db=None,
        output_folder: str = VIDEO_FOLDER,
        temp_dir: str = TEMP_DIR,
        debug_mode: bool = False,
    ):
        self.llm_manager = llm_manager
        self.tts_manager = tts_manager
        self.target_langs = target_langs
        self.db = db
        self.output_folder = output_folder
        self.temp_dir = temp_dir
        self.debug_mode = debug_mode

        self.durations = {}
        self.speaker_info = {}
        self.speaker_refs = {}
        self.global_context = {}
        self.available_pans = [-0.10, 0.10, -0.03, 0.03, -0.17, 0.17]
        self.speaker_pans = {}
        self.abort_event = threading.Event()
        self.monitor = None
        self.synchronizer = SegmentSynchronizer(
            llm_manager=llm_manager, tts_manager=tts_manager, temp_dir=temp_dir, debug_mode=debug_mode
        )

    def _cleanup_debug(self, fname):
        clean_name = os.path.basename(fname)
        d = os.path.join(self.output_folder, "debug_" + os.path.splitext(clean_name)[0])
        if os.path.exists(d):
            shutil.rmtree(d)
        if self.debug_mode:
            os.makedirs(d, exist_ok=True)
        return d

    def _create_script(self, diar, trans):
        raw_script = []
        for t in trans:
            spk = next((d["speaker"] for d in diar if max(t["start"], d["start"]) < min(t["end"], d["end"])), "unknown")
            raw_script.append({**t, "speaker": spk})

        if not raw_script:
            return []

        merged = []
        curr = raw_script[0]
        for i in range(1, len(raw_script)):
            nxt = raw_script[i]
            gap = nxt["start"] - curr["end"]
            if nxt["speaker"] == curr["speaker"] and 0 <= gap < 1.0 and len(curr["text"]) + len(nxt["text"]) < 400:
                curr["text"] = curr["text"].strip() + " " + nxt["text"].strip()
                curr["end"] = nxt["end"]
                if "avg_logprob" in curr and "avg_logprob" in nxt:
                    curr["avg_logprob"] = (curr["avg_logprob"] + nxt["avg_logprob"]) / 2
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)

        script = []
        for i, s in enumerate(merged):
            script.append(
                {
                    **s,
                    "speaker": s["speaker"],
                    "text_en": s["text"].strip(),
                    "avg_logprob": s.get("avg_logprob", 0),
                    "index": i,
                }
            )
        return script

    def _extract_subtitles(self, vpath):
        base_path = os.path.splitext(vpath)[0]
        srt_path = base_path + ".srt"
        content = ""
        if os.path.exists(srt_path):
            logger.info(f"Found external subtitles: {srt_path}")
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read external SRT: {e}")
        if not content:
            try:
                meta = FFmpegWrapper.get_metadata(vpath)
                has_subs = any(s.get("codec_type") == "subtitle" for s in meta.get("streams", []))
                if has_subs:
                    logger.info("Found embedded subtitles. Extracting...")
                    temp_srt = os.path.join(self.temp_dir, "extracted.srt")
                    run_cmd(["ffmpeg", "-i", vpath, "-map", "0:s:0", temp_srt, "-y"], "extract subtitles")
                    if os.path.exists(temp_srt):
                        with open(temp_srt, "r", encoding="utf-8") as f:
                            content = f.read()
            except Exception as e:
                logger.warning(f"Failed to extract embedded subtitles: {e}")
        if content:
            cleaned = clean_srt(content)
            logger.info(f"Loaded reference subtitles ({len(cleaned)} chars)")
            return cleaned
        logger.info("No reference subtitles found.")
        return ""

    def _extract_refs(self, script, vocals_path, ddir):
        logger.info("Extracting Smart Weighted Normalized voice references (v2 - Scored 0-100)")
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

                total = clarity_score + dur_score + pos_score
                scored_candidates.append((s, total))

            if scored_candidates:
                top_3 = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:3]
                final_choice = None

                for cand, score in top_3:
                    tmp_wav = os.path.join(self.temp_dir, f"test_{spk}_{cand['start']}.wav")
                    filt = "highpass=f=100,afftdn=nf=-20,speechnorm=e=10:r=0.0001:l=1"
                    run_cmd(
                        [
                            "ffmpeg",
                            "-i",
                            vocals_path,
                            "-ss",
                            str(cand["start"]),
                            "-t",
                            str(cand["end"] - cand["start"]),
                            "-af",
                            filt,
                            "-ac",
                            "1",
                            "-ar",
                            "24000",
                            tmp_wav,
                            "-y",
                        ],
                        "zcr check",
                    )

                    zcr = measure_zcr(tmp_wav)
                    if zcr > 0.15:
                        if os.path.exists(tmp_wav):
                            os.remove(tmp_wav)
                        continue

                    final_choice = (cand, score, tmp_wav)
                    break

                if final_choice:
                    best_seg, best_score, wav_path = final_choice
                    out = os.path.join(self.temp_dir, f"ref_{spk}.wav")
                    shutil.move(wav_path, out)
                    best_per_speaker[spk] = (best_seg, best_score)
                    self.speaker_refs[spk] = out
                    logger.info(
                        f"Speaker {spk} Selected: Score {best_score} (Dur: {best_seg['end'] - best_seg['start']:.1f}s)"
                    )
                    if ref_debug:
                        shutil.copy(out, os.path.join(ref_debug, f"{spk}.wav"))
                else:
                    logger.warning(f"Speaker {spk}: No clean candidates found after ZCR check.")

        for spk in all_speakers:
            my_best = best_per_speaker.get(spk)
            if not my_best:
                logger.warning(f"  [Speaker {spk}] No candidates found. Voice cloning might fail.")
            elif my_best[1] < 65:
                logger.warning(f"  [Speaker {spk}] Low Score {my_best[1]}. Using best available candidate.")

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

    def process_video(self, vpath: str, task_id: Optional[int] = None):
        if not os.path.isabs(vpath):
            vpath = os.path.join(VIDEO_FOLDER, vpath)

        logger.info(f"=== PROCESSING FILE: {vpath} (Task: {task_id}) ===")
        start_all = time.perf_counter()
        ddir = self._cleanup_debug(vpath)
        seg_dir = os.path.join(ddir, "segments") if self.debug_mode else None

        if seg_dir and os.path.exists(seg_dir):
            shutil.rmtree(seg_dir)
        if seg_dir and self.debug_mode:
            os.makedirs(seg_dir, exist_ok=True)

        self.monitor = ResourceMonitor()
        self.monitor.daemon = True
        self.monitor.start()

        def run_step(name, func, *args):
            # Check for checkpoint if db and task_id provided
            if self.db and task_id:
                cached = self.db.get_step_result(task_id, name)
                if cached is not None:
                    logger.info(f"STEP {name}: Using cached result.")
                    return cached

            logger.info(f"STARTING STEP: {name}")
            t = time.perf_counter()
            res = func(*args)
            self.durations[name] = time.perf_counter() - t
            logger.info(f"COMPLETED STEP: {name} in {humanfriendly.format_timespan(self.durations[name])}")

            if self.db and task_id:
                self.db.save_step_result(task_id, name, "DONE", result_data=res)
            return res

        a_stereo, vocals = run_step("Stage 1: Audio Separation", prep_audio, vpath)
        analysis_data = run_step("Stage 2: Audio Analysis", analyze_audio, vocals)
        diar, trans, audio_durs = analysis_data[0], analysis_data[1], analysis_data[2]
        self.durations.update(audio_durs)

        if not self.llm_manager.llm:
            threading.Thread(target=self.llm_manager.load_model, daemon=True).start()
            logger.info("Waiting for LLM to load...")
            self.llm_manager.ready_event.wait()

        script = self._create_script(diar, trans)
        if self.debug_mode:
            with open(os.path.join(ddir, "script_initial.json"), "w") as f_dbg:
                json.dump(script, f_dbg, indent=2)
        if self.abort_event.is_set():
            return
        ref_subs = self._extract_subtitles(vpath)

        analysis_results = run_step("Stage 3: Global Analysis", self.llm_manager.analyze_script, script, ddir, ref_subs)
        self.global_context = analysis_results["context"]
        self.speaker_info = analysis_results["speakers"]

        run_step("Stage 4: Transcription Correction", self._extract_refs, script, vocals, ddir)

        existing_langs = audio_processor.get_audio_languages(vpath)
        if existing_langs:
            logger.info(f"Existing audio languages: {', '.join(existing_langs)}")

        iso_map = {
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

        all_audio_tracks = []
        for lang in self.target_langs:
            if lang.lower() in existing_langs or iso_map.get(lang.lower()) in existing_langs:
                logger.info(f"--- SKIPPING {lang}: Language already exists in source video ---")
                continue

            logger.info(f"--- STARTING PRODUCTION FOR LANGUAGE: {lang} ---")

            # 1. Draft Translation (Batch)
            t_start_draft = time.perf_counter()
            draft_segments = self.llm_manager.generate_drafts(script, lang, self.speaker_info, self.global_context)
            self.durations[f"Stage 5a: Draft Translation ({lang})"] = time.perf_counter() - t_start_draft

            # 2. Parallel Refinement (Sync TTS Loop)
            res = []
            t_start_sync = time.perf_counter()

            # Use max_workers=3 to balance GPU usage (TTS) and CPU usage (FFmpeg/LLM)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.synchronizer.process_segment, seg, lang, vocals, script, self.global_context): seg[
                        "index"
                    ]
                    for seg in draft_segments
                }

                completed_count = 0
                total_segments = len(draft_segments)

                for future in as_completed(futures):
                    if self.abort_event.is_set():
                        executor.shutdown(wait=False)
                        raise RuntimeError("Processing aborted via event")

                    idx = futures[future]
                    try:
                        seg_result = future.result()
                        if seg_result:
                            # Add back metadata needed for mixing
                            seg_result["start"] = script[idx]["start"]
                            seg_result["end"] = script[idx]["end"]
                            seg_result["speaker"] = script[idx]["speaker"]
                            seg_result["index"] = idx
                            res.append(seg_result)
                            completed_count += 1
                            if completed_count % 5 == 0:
                                logger.info(f"  [Progress] {completed_count}/{total_segments} segments done.")
                    except Exception as e:
                        logger.error(f"Segment {idx} failed completely: {e}")

            self.durations[f"Stage 5b: Sync Production ({lang})"] = time.perf_counter() - t_start_sync

            if self.debug_mode:
                # Debug dump for sync results
                with open(os.path.join(ddir, f"sync_results_{lang}.json"), "w") as f_dbg:
                    json.dump(res, f_dbg, indent=2, default=str)

            # 3. Mastering & Mixing preparation
            res.sort(key=lambda x: x["index"])

            final_audio_segments = []
            last_end_time = 0

            for item in res:
                start = item["start"]
                # Use the refined duration from synchronizer
                dur = item["duration"]
                path = item["audio_path"]

                if start < last_end_time:
                    shift = last_end_time - start
                    # Only shift if overlap is significant, otherwise mix/crossfade handles it
                    if shift > 0.05:
                        start = last_end_time

                final_path = os.path.join(self.temp_dir, f"final_{lang}_{item['index']}.wav")

                # Trim silence at ends to be clean
                audio_processor.trim_and_pad_silence(path, dur)

                # Calculate final speed factor if needed (Synchronizer should have handled it mostly)
                target_dur = item["end"] - item["start"]
                actual_dur = dur

                speed_factor = 1.0
                # Gentle speedup if still too long (safety net)
                if actual_dur > target_dur + 0.2:
                    speed_factor = min(actual_dur / target_dur, 1.20)

                self._apply_mastering_and_speed(path, final_path, item["speaker"], speed_factor)

                final_audio_segments.append((final_path, start, actual_dur / speed_factor))
                last_end_time = start + (actual_dur / speed_factor)

            final_a = os.path.join(self.temp_dir, f"final_{lang}.ac3")
            run_step(f"Stage 6: Final Mix ({lang})", mix_audio, a_stereo, final_audio_segments, final_a)
            all_audio_tracks.append((final_a, lang, LANG_MAP.get(lang, lang)))

        if self.db and task_id:
            self.db.save_step_result(task_id, "Stage 5: Production", "DONE", result_data={"langs": self.target_langs})

        if all_audio_tracks:
            ext = os.path.splitext(vpath)[1] or ".mkv"
            final_video = os.path.join(self.temp_dir, f"final_muxed{ext}")
            run_step("Stage 7: Muxing", FFmpegWrapper.mux_video, vpath, all_audio_tracks, final_video)
            logger.info(f"Replacing original file: {vpath}")
            shutil.move(final_video, vpath)

        if self.monitor:
            self.monitor.stop()
        avg_tps = (
            self.llm_manager.llm_stats["tokens"] / self.llm_manager.llm_stats["time"]
            if self.llm_manager.llm_stats["time"] > 0
            else 0
        )
        self._print_report(vpath, time.perf_counter() - start_all, avg_tps)

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
