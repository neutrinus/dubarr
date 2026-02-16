import os
import json
import logging
import time
import threading
import queue
import sys
import humanfriendly
import shutil
import subprocess

# Import local modules
from config import (
    DEBUG_MODE,
    setup_logging,
    LANG_MAP,
    VIDEO_FOLDER,
    OUTPUT_FOLDER,
    TEMP_DIR,
    MODEL_PATH,
    DEVICE_LLM,
    DEVICE_AUDIO,
    USE_LOCK,
    TARGET_LANGS,
)
from utils import clean_srt, measure_zcr, run_cmd
from monitor import ResourceMonitor
from llm_engine import LLMManager
from tts_manager import TTSManager
import audio_processor

setup_logging()


class AIDubber:
    def __init__(self):
        self.video_folder = VIDEO_FOLDER
        self.output_folder = OUTPUT_FOLDER
        self.temp_dir = TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        self.target_langs = TARGET_LANGS
        self.debug_mode = DEBUG_MODE

        self.device_llm = DEVICE_LLM
        self.device_audio = DEVICE_AUDIO
        self.inference_lock = threading.Lock() if USE_LOCK else None

        logging.info("--- DUBARR CONFIGURATION ---")
        logging.info(f"LLM Device:   {self.device_llm}")
        logging.info(f"Audio Device: {self.device_audio}")
        logging.info(f"Locking:      {'ENABLED' if USE_LOCK else 'DISABLED'}")
        logging.info("----------------------------")

        self.durations = {}
        self.speaker_info = {}
        self.speaker_refs = {}
        self.global_context = {}
        self.available_pans = [-0.10, 0.10, -0.03, 0.03, -0.17, 0.17]
        self.speaker_pans = {}
        self.abort_event = threading.Event()
        self.llm_manager = LLMManager(
            model_path=MODEL_PATH,
            device=self.device_llm,
            inference_lock=self.inference_lock,
            debug_mode=self.debug_mode,
            target_langs=self.target_langs,
        )
        self.tts_manager = TTSManager(
            device=self.device_audio,
            inference_lock=self.inference_lock,
            temp_dir=self.temp_dir,
            speaker_refs=self.speaker_refs,
            abort_event=self.abort_event,
        )
        self.monitor = None

    def _cleanup_debug(self, fname):
        d = os.path.join(self.output_folder, "debug_" + os.path.splitext(fname)[0])
        if os.path.exists(d):
            shutil.rmtree(d)
        if self.debug_mode:
            os.makedirs(d, exist_ok=True)
        return d

    def _create_script(self, diar, trans):
        script = []
        for t in trans:
            spk = next((d["speaker"] for d in diar if max(t["start"], d["start"]) < min(t["end"], d["end"])), "unknown")
            script.append({**t, "speaker": spk, "text_en": t["text"], "avg_logprob": t.get("avg_logprob", 0)})
        return script

    def _extract_subtitles(self, vpath):
        base_path = os.path.splitext(vpath)[0]
        srt_path = base_path + ".srt"
        content = ""
        if os.path.exists(srt_path):
            logging.info(f"Found external subtitles: {srt_path}")
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logging.warning(f"Failed to read external SRT: {e}")
        if not content:
            try:
                res = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "s",
                        "-show_entries",
                        "stream=index",
                        "-of",
                        "csv=p=0",
                        vpath,
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.stdout.strip():
                    logging.info("Found embedded subtitles. Extracting...")
                    temp_srt = os.path.join(self.temp_dir, "extracted.srt")
                    run_cmd(["ffmpeg", "-i", vpath, "-map", "0:s:0", temp_srt, "-y"], "extract subtitles")
                    if os.path.exists(temp_srt):
                        with open(temp_srt, "r", encoding="utf-8") as f:
                            content = f.read()
            except Exception as e:
                logging.warning(f"Failed to extract embedded subtitles: {e}")
        if content:
            cleaned = clean_srt(content)
            logging.info(f"Loaded reference subtitles ({len(cleaned)} chars)")
            return cleaned
        logging.info("No reference subtitles found.")
        return ""

    def _extract_refs(self, script, vocals_path, ddir):
        logging.info("Extracting Smart Weighted Normalized voice references (v2 - Scored 0-100)")
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
                    logging.info(
                        f"Speaker {spk} Selected: Score {best_score} (Dur: {best_seg['end'] - best_seg['start']:.1f}s)"
                    )
                    if ref_debug:
                        shutil.copy(out, os.path.join(ref_debug, f"{spk}.wav"))
                else:
                    logging.warning(f"Speaker {spk}: No clean candidates found after ZCR check.")

        for spk in all_speakers:
            my_best = best_per_speaker.get(spk)
            if not my_best:
                logging.warning(f"  [Speaker {spk}] No candidates found. Voice cloning might fail.")
            elif my_best[1] < 65:
                logging.warning(f"  [Speaker {spk}] Low Score {my_best[1]}. Using best available candidate.")

    def _audio_postprocessor(self, q_in, results):
        while not self.abort_event.is_set():
            try:
                task = q_in.get(timeout=2)
            except queue.Empty:
                continue
            if task is None:
                break
            item = task["item"]
            raw = task["raw_path"]
            max_dur = task["max_dur"]
            final = os.path.join(self.temp_dir, f"tts_{item['index']}.wav")
            try:
                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    raw,
                ]
                raw_dur = float(subprocess.check_output(cmd).strip())
                if raw_dur > max_dur:
                    tmp_cut = raw + ".cut.wav"
                    subprocess.run(
                        ["ffmpeg", "-i", raw, "-t", str(max_dur), "-c", "copy", tmp_cut, "-y"], capture_output=True, check=True
                    )
                    os.replace(tmp_cut, raw)
                audio_processor.trim_silence(raw)
                out_dur_str = subprocess.check_output(cmd).strip()
                actual_dur = float(out_dur_str)
                target_dur = item["end"] - item["start"]
                speed_factor = min(actual_dur / target_dur, 1.25) if actual_dur > target_dur else 1.0
                self._apply_mastering_and_speed(raw, final, item["speaker"], speed_factor)
                results.append((final, item["start"], actual_dur / speed_factor))
                logging.info(f"  [ID: {item['index']}] POST-PROC DONE. (Spd: {speed_factor:.2f}x)")
            except Exception as e:
                logging.error(f"Postproc failed {item['index']}: {e}")
            q_in.task_done()

    def _apply_mastering_and_speed(self, r, f, spk, speed):
        p = self.speaker_pans.get(spk)
        if p is None:
            info = self.speaker_info.get(spk, {})
            name = info.get("name", "").lower()
            desc = info.get("desc", "").lower()
            is_narrator = any(x in name or x in desc for x in ["narrator", "lektor", "narratorka"])
            if is_narrator:
                p = 0.0
                logging.info(f"Speaker {spk} identified as Narrator. Centering audio.")
            else:
                p = self.available_pans.pop(0) if self.available_pans else 0.0
            self.speaker_pans[spk] = p
        info = self.speaker_info.get(spk, {})
        is_narrator = any(
            x in info.get("name", "").lower() or x in info.get("desc", "").lower()
            for x in ["narrator", "lektor", "narratorka"]
        )
        echo = "aecho=0.8:0.9:10:0.2," if not is_narrator else ""
        filt = f"highpass=f=60,{echo}speechnorm=e=6:r=0.0001:l=1,pan=stereo|c0={1.0 - max(0, p):.2f}*c0|c1={1.0 + min(0, p):.2f}*c0,atempo={speed}"
        subprocess.run(["ffmpeg", "-i", r, "-af", filt, f, "-y"], capture_output=True)

    def process_video(self, f):
        vpath = os.path.join(self.video_folder, f)
        logging.info(f"=== PROCESSING FILE: {f} for languages: {','.join(self.target_langs)} ===")
        start_all = time.perf_counter()
        ddir = self._cleanup_debug(f)
        seg_dir = os.path.join(ddir, "segments") if self.debug_mode else None
        if os.path.exists(seg_dir):
            shutil.rmtree(seg_dir)
        if self.debug_mode:
            os.makedirs(seg_dir, exist_ok=True)
        monitor_state = {}
        self.monitor = ResourceMonitor(monitor_state)
        self.monitor.daemon = True
        self.monitor.start()

        def step(name, func, *args):
            logging.info(f"STARTING STEP: {name}")
            t = time.perf_counter()
            res = func(*args)
            self.durations[name] = time.perf_counter() - t
            logging.info(f"COMPLETED STEP: {name} in {humanfriendly.format_timespan(self.durations[name])}")
            return res

        a_stereo, vocals = step("1. Audio Separation (Demucs)", audio_processor.prep_audio, vpath)
        diar, trans, audio_durs = step("2. Audio Analysis (Whisper/Diarization)", audio_processor.analyze_audio, vocals)
        self.durations.update(audio_durs)

        # Load LLM only after memory-intensive Audio Analysis is done
        if not self.llm_manager.llm:
            threading.Thread(target=self.llm_manager.load_model, daemon=True).start()
            logging.info("Waiting for LLM to load...")
            self.llm_manager.ready_event.wait()

        script = self._create_script(diar, trans)
        if self.debug_mode:
            with open(os.path.join(ddir, "script_initial.json"), "w") as f_dbg:
                json.dump(script, f_dbg, indent=2)
        if self.abort_event.is_set():
            return
        ref_subs = self._extract_subtitles(vpath)

        # LLM Phase 1-3
        analysis_results = step(
            "3. LLM Enhancement (Context/Speakers/ASR Fix)", self.llm_manager.analyze_script, script, ddir, ref_subs
        )
        self.global_context = analysis_results["context"]
        self.speaker_info = analysis_results["speakers"]

        step("4. Voice Reference Extraction", self._extract_refs, script, vocals, ddir)

        for lang in self.target_langs:
            logging.info(f"--- STARTING PRODUCTION FOR LANGUAGE: {lang} ---")
            q_text = queue.Queue(maxsize=15)
            q_audio = queue.Queue(maxsize=10)
            monitor_state["q_text"] = q_text
            monitor_state["q_audio"] = q_audio
            res = []
            draft_translations = []
            final_translations = []

            p_th = threading.Thread(
                target=self.llm_manager.translation_producer,
                args=(script, lang, q_text, self.speaker_info, self.global_context, draft_translations, final_translations),
                daemon=True,
            )
            tts_th = threading.Thread(
                target=self.tts_manager.tts_worker,
                args=(lang, q_text, q_audio, [], vocals, script, self.durations),
                daemon=True,
            )
            post_th = threading.Thread(target=self._audio_postprocessor, args=(q_audio, res), daemon=True)

            p_th.start()
            tts_th.start()
            post_th.start()

            while p_th.is_alive() or tts_th.is_alive() or post_th.is_alive():
                if self.abort_event.is_set():
                    sys.exit(1)
                time.sleep(1.0)

            p_th.join()
            tts_th.join()
            post_th.join()

            if self.debug_mode:
                with open(os.path.join(ddir, f"translations_draft_{lang}.json"), "w") as f_out:
                    json.dump(draft_translations, f_out, indent=2, ensure_ascii=False)
                with open(os.path.join(ddir, f"translations_final_{lang}.json"), "w") as f_out:
                    json.dump(final_translations, f_out, indent=2, ensure_ascii=False)
                for path, _, _ in res:
                    shutil.copy(path, os.path.join(seg_dir, f"{lang}_{os.path.basename(path)}"))
            res.sort(key=lambda x: x[1])
            final_a = os.path.join(self.temp_dir, f"final_{lang}.ac3")
            step(f"6. Final Mix ({lang})", audio_processor.mix_audio, a_stereo, res, final_a)
            step(
                f"7. Muxing ({lang})",
                audio_processor.mux_video,
                vpath,
                final_a,
                lang,
                os.path.join(self.output_folder, f"dub_{lang}_{f}"),
                LANG_MAP.get(lang, lang),
            )

        if self.monitor:
            self.monitor.stop()

        # Stats and Cleanup
        avg_tps = (
            self.llm_manager.llm_stats["tokens"] / self.llm_manager.llm_stats["time"]
            if self.llm_manager.llm_stats["time"] > 0
            else 0
        )
        self._print_report(f, time.perf_counter() - start_all, avg_tps)

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
        logging.info("\n".join(rep))

    def run(self):
        try:
            for f in os.listdir(self.video_folder):
                if f.endswith((".mkv", ".mp4")):
                    self.process_video(f)
        except Exception:
            logging.exception("FATAL ERROR in main loop")


if __name__ == "__main__":
    try:
        AIDubber().run()
    except Exception:
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
        logging.exception("FATAL ERROR")
        raise
