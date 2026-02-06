import os
import json
import logging
import time
import threading
import queue
import torch
import gc
import sys
import humanfriendly
import shutil
from typing import List, Dict, Tuple, Optional

# Import local modules
from config import (
    DEBUG_MODE, LOG_LEVEL, setup_logging, LANG_MAP, 
    VIDEO_FOLDER, OUTPUT_FOLDER, TEMP_DIR, MODEL_PATH,
    GPU_LLM, GPU_AUDIO, TARGET_LANGS, HF_TOKEN
)
from prompts import (
    T_ANALYSIS, T_ED, T_TRANS_SYSTEM, T_CRITIC_SYSTEM, 
    T_TRANS, T_CRITIC, T_SHORTEN
)
from utils import (
    parse_json, clean_srt, measure_zcr, count_syllables, 
    clean_output, run_cmd
)
from monitor import ResourceMonitor
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
        
        self.gpu_llm = GPU_LLM
        self.gpu_audio = GPU_AUDIO
        
        self.durations = {}
        self.llm_stats = {"tokens": 0, "time": 0}
        self.speaker_info = {}
        self.speaker_refs = {}
        self.global_context = {}
        self.available_pans = [-0.10, 0.10, -0.03, 0.03, -0.17, 0.17]
        self.speaker_pans = {}
        self.abort_event = threading.Event()
        self.llm = None
        self.monitor = None
        self.llm_ready = threading.Event()

    def _cleanup_debug(self, fname):
        d = os.path.join(self.output_folder, "debug_" + os.path.splitext(fname)[0])
        if os.path.exists(d):
            shutil.rmtree(d)
        if self.debug_mode:
            os.makedirs(d, exist_ok=True)
        return d

    def _load_model(self):
        logging.info(f"Background: Loading LLM on Dedicated GPU {self.gpu_llm}...")
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                main_gpu=self.gpu_llm,
                n_ctx=4096, 
                n_batch=512,
                n_threads=4,
                flash_attn=True,
                verbose=False
            )
            logging.info("Background: LLM Loaded and ready.")
            self.llm_ready.set()
        except Exception as e:
            logging.error(f"Background: LLM Load Failed: {e}")
            self.abort_event.set()

    def _create_script(self, diar, trans):
        script = []
        for t in trans:
            spk = next((d["speaker"] for d in diar if max(t["start"], d["start"]) < min(t["end"], d["end"])), "unknown")
            script.append({**t, "speaker": spk, "text_en": t["text"], "avg_logprob": t.get("avg_logprob", 0)})
        return script

    def _llm_phase(self, script, ddir, subtitles=""):
        if not self.llm:
            raise RuntimeError("LLM model not loaded!")
        
        max_overview_lines = 400 
        if len(script) > max_overview_lines:
            logging.warning(f"Script too long ({len(script)} lines). Truncating overview for Profiler.")
            mid = len(script) // 2
            ov_subset = script[mid-200 : mid+200]
            ov = "\n".join([f"{s['speaker']}: {s['text_en']}" for s in ov_subset])
        else:
            ov = "\n".join([f"{s['speaker']}: {s['text_en']}" for s in script])

        lang_names = [LANG_MAP.get(l, l) for l in self.target_langs]
        logging.info("LLM Stage 1 & 2 Combined: Full Analysis (Context + Profiling)")
        
        t0 = time.perf_counter()
        res = self.llm(T_ANALYSIS.format(overview=ov, langs=", ".join(lang_names), subtitles=subtitles), max_tokens=2000, stop=["<|im_end|>"])
        dt = time.perf_counter() - t0
        toks = res.get("usage", {}).get("completion_tokens", 0)
        self.llm_stats["tokens"] += toks; self.llm_stats["time"] += dt

        analysis = parse_json(res["choices"][0]["text"])
        self.global_context = {k: v for k, v in analysis.items() if k != "speakers"}
        self.speaker_info = analysis.get("speakers", {})

        logging.info("LLM Stage 3: ASR Correction (Editor - Diff Only)")
        chunk_sz = 100
        glossary_str = json.dumps(self.global_context.get('glossary', {}))
        for i in range(0, len(script), chunk_sz):
            chunk = script[i:i+chunk_sz]
            txt = "\n".join([f"L_{i+j}: {s['text_en']}" for j, s in enumerate(chunk)])
            t0 = time.perf_counter()
            res = self.llm(T_ED.format(glossary=glossary_str, subtitles=subtitles, txt=txt), max_tokens=2000, temperature=0.0, stop=["<|im_end|>"])
            dt = time.perf_counter() - t0
            toks = res.get("usage", {}).get("completion_tokens", 0)
            self.llm_stats["tokens"] += toks; self.llm_stats["time"] += dt
            
            try:
                corrections = parse_json(res["choices"][0]["text"])
                if isinstance(corrections, dict):
                    for idx_str, new_text in corrections.items():
                        try:
                            idx = int(idx_str.replace("L_", ""))
                            if 0 <= idx < len(script):
                                if script[idx]["text_en"] != new_text and self.debug_mode:
                                    logging.info(f"ASR Fix L_{idx}: '{script[idx]['text_en']}' -> '{new_text}'")
                                script[idx]["text_en"] = new_text
                        except ValueError:
                            continue
            except Exception as e:
                logging.warning(f"Failed to parse ASR corrections for chunk starting at {i}: {e}")

        if self.debug_mode:
            with open(os.path.join(ddir, "context.json"), "w") as f:
                json.dump(self.global_context, f, indent=2)
            with open(os.path.join(ddir, "speakers.json"), "w") as f:
                json.dump(self.speaker_info, f, indent=2)
            with open(os.path.join(ddir, "script_fixed.json"), "w") as f:
                json.dump(script, f, indent=2)

    def _extract_subtitles(self, vpath):
        base_path = os.path.splitext(vpath)[0]
        srt_path = base_path + ".srt"
        content = ""
        if os.path.exists(srt_path):
            logging.info(f"Found external subtitles: {srt_path}")
            try:
                with open(srt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logging.warning(f"Failed to read external SRT: {e}")
        if not content:
            try:
                res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                                     "stream=index", "-of", "csv=p=0", vpath], capture_output=True, text=True)
                if res.stdout.strip():
                    logging.info("Found embedded subtitles. Extracting...")
                    temp_srt = os.path.join(self.temp_dir, "extracted.srt")
                    run_cmd(["ffmpeg", "-i", vpath, "-map", "0:s:0", temp_srt, "-y"], "extract subtitles")
                    if os.path.exists(temp_srt):
                        with open(temp_srt, 'r', encoding='utf-8') as f:
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
        if ref_debug: os.makedirs(ref_debug, exist_ok=True)
        
        bad_markers = ["[", "]", "(", ")", "music", "laughter", "scream", "explosion", "sound", "noise", "intro", "theme"]
        total_dur = script[-1]["end"] if script else 0
        
        best_per_speaker = {}
        all_speakers = set(s["speaker"] for s in script if s["speaker"] != "unknown")
        
        for spk in all_speakers:
            candidates = [s for s in script if s["speaker"] == spk]
            scored_candidates = []
            
            for s in candidates:
                txt = s["text_en"].lower()
                if any(x in txt for x in bad_markers): continue
                dur = s["end"] - s["start"]
                if dur < 2.0 or dur > 20.0: continue
                
                clarity_score = 50
                conf = s.get("avg_logprob", -0.5)
                if conf < -0.4: clarity_score -= 10
                if conf < -0.7: clarity_score -= 15
                
                nsp = s.get("no_speech_prob", 0)
                if nsp > 0.1: clarity_score -= 10
                if nsp > 0.3: clarity_score -= 20
                
                cr = s.get("compression_ratio", 1.2)
                if cr > 1.8: clarity_score = 0 
                if cr < 0.6: clarity_score -= 10 
                
                words = len(s["text_en"].split())
                wps = words / dur
                if wps < 0.5 or wps > 4.5: clarity_score -= 15
                
                if clarity_score < 0: clarity_score = 0
                
                dur_score = 0
                if 6.0 <= dur <= 14.0: dur_score = 40
                elif 4.0 <= dur < 6.0: dur_score = 25
                elif 14.0 < dur <= 18.0: dur_score = 20
                else: dur_score = 5
                
                pos_score = 0
                if total_dur > 0:
                    rel_pos = s["start"] / total_dur
                    if 0.15 < rel_pos < 0.85: pos_score = 10
                    elif 0.05 < rel_pos < 0.95: pos_score = 5
                
                total = clarity_score + dur_score + pos_score
                scored_candidates.append((s, total))
            
            if scored_candidates:
                top_3 = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:3]
                final_choice = None
                
                for cand, score in top_3:
                    tmp_wav = os.path.join(self.temp_dir, f"test_{spk}_{cand['start']}.wav")
                    filt = "highpass=f=100,afftdn=nf=-20,speechnorm=e=10:r=0.0001:l=1"
                    run_cmd(["ffmpeg", "-i", vocals_path, "-ss", str(cand["start"]), "-t", str(cand["end"]-cand["start"]), "-af", filt, "-ac", "1", "-ar", "22050", tmp_wav, "-y"], "zcr check")
                    
                    zcr = measure_zcr(tmp_wav)
                    if zcr > 0.15: 
                        if os.path.exists(tmp_wav): os.remove(tmp_wav)
                        continue 
                    
                    final_choice = (cand, score, tmp_wav) 
                    break
                
                if final_choice:
                    best_seg, best_score, wav_path = final_choice
                    out = os.path.join(self.temp_dir, f"ref_{spk}.wav")
                    shutil.move(wav_path, out)
                    best_per_speaker[spk] = (best_seg, best_score)
                    self.speaker_refs[spk] = out
                    logging.info(f"Speaker {spk} Selected: Score {best_score} (Dur: {best_seg['end']-best_seg['start']:.1f}s)")
                    if ref_debug: shutil.copy(out, os.path.join(ref_debug, f"{spk}.wav"))
                else:
                    logging.warning(f"Speaker {spk}: No clean candidates found after ZCR check.")

        for spk in all_speakers:
            my_best = best_per_speaker.get(spk)
            info = self.speaker_info.get(spk, {})
            name = info.get("name", "").lower(); desc = info.get("desc", "").lower()
            is_female = any(x in desc or x in name for x in ["female", "woman", "lady", "girl", "kobieta", "pani"])
            
            if not my_best or my_best[1] < 65:
                self.speaker_refs[spk] = "Daisy Morgan" if is_female else "Damien Sayre"
                reason = "No candidates" if not my_best else f"Low Score {my_best[1]}"
                logging.warning(f"  [Speaker {spk}] Fallback to GENERIC ({self.speaker_refs[spk]}) - {reason}")

    def _producer(self, script, lang, q_text, spk_info, glob_ctx, draft_list, final_list):
        t_trans_total = 0; t_critic_total = 0
        total_tokens = 0
        try:
            if not self.llm: raise RuntimeError("LLM model not loaded in producer!")
            summary = glob_ctx.get('summary', ''); glossary = glob_ctx.get('glossary', {}); lang_name = LANG_MAP.get(lang, lang)
            valid_indices = [idx for idx, s in enumerate(script) if not any(x in s["text_en"].lower() for x in ["[", "(", "laughter", "screams"])]
            
            for b_start in range(0, len(valid_indices), 10):
                if self.abort_event.is_set(): return
                batch_indices = valid_indices[b_start:b_start+10]
                json_batch = [{"id": idx, "speaker": spk_info.get(script[idx]['speaker'], {}).get('name', script[idx]['speaker']), "text": script[idx]['text_en']} for idx in batch_indices]
                trans_prompt = T_TRANS.format(system_prompt=T_TRANS_SYSTEM.format(lang_name=lang_name, glossary=json.dumps(glossary, ensure_ascii=False)), json_input=json.dumps(json_batch, indent=2))
                
                t_start = time.perf_counter()
                res = self.llm(trans_prompt, max_tokens=1500, temperature=0.0, stop=["<|im_end|>"])
                dt = time.perf_counter() - t_start
                tokens = res.get("usage", {}).get("completion_tokens", 0)
                total_tokens += tokens; t_trans_total += dt
                self.llm_stats["tokens"] += tokens; self.llm_stats["time"] += dt

                parsed_trans = parse_json(res["choices"][0]["text"])
                trans_map = {item["id"]: clean_output(item["text"], self.target_langs) for item in parsed_trans.get("translations", []) if "id" in item}
                
                json_critic = []; final_batch_map = {}
                for idx in batch_indices:
                    txt = trans_map.get(idx, script[idx]["text_en"])
                    draft_list.append({"index": idx, "text": txt, "speaker": script[idx]["speaker"], "start": script[idx]["start"], "end": script[idx]["end"]})
                    syl_draft = count_syllables(txt, lang)
                    if syl_draft <= 3: final_batch_map[idx] = txt
                    else: json_critic.append({"id": idx, "original": script[idx]["text_en"], "draft": txt})
                
                if json_critic:
                    critic_prompt = T_CRITIC.format(system_prompt=T_CRITIC_SYSTEM.format(summary=summary, glossary=json.dumps(glossary, ensure_ascii=False)), json_input=json.dumps(json_critic, indent=2))
                    t_start = time.perf_counter()
                    res_crit = self.llm(critic_prompt, max_tokens=1500, temperature=0.0, stop=["<|im_end|>"])
                    dt = time.perf_counter() - t_start
                    tokens = res_crit.get("usage", {}).get("completion_tokens", 0)
                    total_tokens += tokens; t_critic_total += dt
                    self.llm_stats["tokens"] += tokens; self.llm_stats["time"] += dt

                    parsed_crit = parse_json(res_crit["choices"][0]["text"])
                    for item in parsed_crit.get("final_translations", []):
                        if "id" in item: final_batch_map[item["id"]] = clean_output(item.get("final_text"), self.target_langs) or trans_map.get(item["id"])

                for idx in batch_indices:
                    txt = final_batch_map.get(idx, trans_map.get(idx, script[idx]["text_en"]))
                    syl_orig = count_syllables(script[idx]['text_en'], "en")
                    syl_final = count_syllables(txt, lang)
                    if syl_final > syl_orig * 1.5:
                        t0 = time.perf_counter()
                        short_res = self.llm(T_SHORTEN.format(original=script[idx]['text_en'], text=txt), max_tokens=150, temperature=0.0, stop=["<|im_end|>"])
                        dt = time.perf_counter() - t0
                        toks = short_res.get("usage", {}).get("completion_tokens", 0)
                        self.llm_stats["tokens"] += toks; self.llm_stats["time"] += dt
                        short_txt = clean_output(parse_json(short_res["choices"][0]["text"]).get("final_text", txt), self.target_langs)
                        if len(short_txt) > 2: txt = short_txt
                    final_item = {"index": idx, "text": txt, "speaker": script[idx]["speaker"], "start": script[idx]["start"], "end": script[idx]["end"]}
                    final_list.append(final_item)
                    q_text.put(final_item)
            self.durations[f"5a. LLM Translation ({lang})"] = t_trans_total; self.durations[f"5b. LLM Critique ({lang})"] = t_critic_total
        except Exception as e:
            logging.exception("Producer thread failed"); self.abort_event.set()
        finally:
            q_text.put(None)

    def _tts_worker(self, lang, q_in, q_out, translations):
        try:
            from TTS.api import TTS
            os.environ["COQUI_TOS_AGREED"] = "1"
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(f"cuda:{self.gpu_audio}")
            available_coqui_speakers = []
            try:
                if hasattr(tts, 'speakers'): available_coqui_speakers = tts.speakers
                elif hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model'): available_coqui_speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
            except: pass
            if not available_coqui_speakers: available_coqui_speakers = ["Claribel Dervla", "Damien Sayre", "Daisy Morgan", "Ana Florence"]
            generic_female_speaker = None; generic_male_speaker = None
            if available_coqui_speakers:
                female_priorities = ["Claribel Dervla", "Daisy Morgan", "Ana Florence"]
                male_priorities = ["Damien Sayre", "Damien Black", "Baldur Sanjin"]
                for p in female_priorities:
                    if p in available_coqui_speakers: generic_female_speaker = p; break
                for p in male_priorities:
                    if p in available_coqui_speakers: generic_male_speaker = p; break
                if not generic_female_speaker: generic_female_speaker = available_coqui_speakers[0]
                if not generic_male_speaker: generic_male_speaker = available_coqui_speakers[-1]

            while not self.abort_event.is_set():
                try: item = q_in.get(timeout=2)
                except queue.Empty: continue
                if item is None: break
                translations.append(item); ref = self.speaker_refs.get(item["speaker"])
                if not ref:
                    q_in.task_done(); continue
                clean_text = item["text"].strip()
                if len(clean_text.split()) < 3 and not clean_text.endswith("..."): clean_text = clean_text + "..."
                raw = os.path.join(self.temp_dir, f"raw_{item['index']}.wav")
                syl_count = count_syllables(clean_text, lang)
                max_allowed_dur = max((syl_count * 0.4) + 1.5, (item["end"] - item["start"]) + 1.5)
                
                try:
                    tts_args = {"text": clean_text, "language": lang, "file_path": raw, "temperature": 0.75, "repetition_penalty": 1.2, "top_p": 0.8, "top_k": 50, "speed": 1.0}
                    if os.path.exists(ref): tts_args["speaker_wav"] = ref; voice_type = "CLONED"
                    else:
                        info = self.speaker_info.get(item["speaker"], {})
                        desc = info.get("desc", "").lower(); name = info.get("name", "").lower()
                        is_female = any(x in desc or x in name for x in ["female", "woman", "lady", "girl", "kobieta", "pani"])
                        fallback = generic_female_speaker if is_female else generic_male_speaker
                        tts_args["speaker"] = fallback if fallback else available_coqui_speakers[0]; voice_type = f"GENERIC_{tts_args['speaker']}"
                    
                    t0 = time.perf_counter()
                    tts.tts_to_file(**tts_args)
                    zcr = measure_zcr(raw)
                    if zcr > 0.25 and "speaker_wav" in tts_args:
                        logging.warning(f"[ID: {item['index']}] High ZCR ({zcr:.3f}). Retrying with GENERIC.")
                        tts_args.pop("speaker_wav")
                        info = self.speaker_info.get(item["speaker"], {})
                        desc = info.get("desc", "").lower(); name = info.get("name", "").lower()
                        is_female = any(x in desc or x in name for x in ["female", "woman", "lady", "girl", "kobieta", "pani"])
                        fb = generic_female_speaker if is_female else generic_male_speaker
                        tts_args["speaker"] = fb if fb else available_coqui_speakers[0]
                        tts.tts_to_file(**tts_args); voice_type = f"RETRY_{tts_args['speaker']}"
                    logging.debug(f"[ID: {item['index']}] TTS Done in {time.perf_counter() - t0:.2f}s ({voice_type})")
                    q_out.put({"item": item, "raw_path": raw, "max_dur": max_allowed_dur, "voice_type": voice_type, "lang": lang})
                except Exception as e: logging.error(f"TTS Gen failed {item['index']}: {e}")
                q_in.task_done()
            del tts; gc.collect(); torch.cuda.empty_cache()
        except Exception as e: logging.exception("TTS Worker failed"); self.abort_event.set()
        finally: q_out.put(None)

    def _audio_postprocessor(self, q_in, results):
        while not self.abort_event.is_set():
            try: task = q_in.get(timeout=2)
            except queue.Empty: continue
            if task is None: break
            item = task["item"]; raw = task["raw_path"]; max_dur = task["max_dur"]; final = os.path.join(self.temp_dir, f"tts_{item['index']}.wav")
            try:
                raw_dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", raw]).strip())
                if raw_dur > max_dur:
                    tmp_cut = raw + ".cut.wav"
                    subprocess.run(["ffmpeg", "-i", raw, "-t", str(max_dur), "-c", "copy", tmp_cut, "-y"], capture_output=True, check=True)
                    os.replace(tmp_cut, raw)
                audio_processor.trim_silence(raw)
                out_dur_str = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", raw]).strip()
                actual_dur = float(out_dur_str); target_dur = item["end"] - item["start"]; speed_factor = min(actual_dur / target_dur, 1.25) if actual_dur > target_dur else 1.0
                self._apply_mastering_and_speed(raw, final, item["speaker"], speed_factor)
                results.append((final, item["start"], actual_dur / speed_factor))
                logging.info(f"  [ID: {item['index']}] POST-PROC DONE. (Spd: {speed_factor:.2f}x)")
            except Exception as e: logging.error(f"Postproc failed {item['index']}: {e}")
            q_in.task_done()

    def _apply_mastering_and_speed(self, r, f, spk, speed):
        p = self.speaker_pans.get(spk)
        if p is None:
            info = self.speaker_info.get(spk, {}); name = info.get("name", "").lower(); desc = info.get("desc", "").lower()
            is_narrator = any(x in name or x in desc for x in ["narrator", "lektor", "narratorka"])
            if is_narrator: p = 0.0; logging.info(f"Speaker {spk} identified as Narrator. Centering audio.")
            else: p = self.available_pans.pop(0) if self.available_pans else 0.0
            self.speaker_pans[spk] = p
        info = self.speaker_info.get(spk, {}); is_narrator = any(x in info.get("name", "").lower() or x in info.get("desc", "").lower() for x in ["narrator", "lektor", "narratorka"])
        echo = "aecho=0.8:0.9:10:0.2," if not is_narrator else ""
        filt = f"highpass=f=60,{echo}speechnorm=e=6:r=0.0001:l=1,pan=stereo|c0={1.0-max(0,p):.2f}*c0|c1={1.0+min(0,p):.2f}*c0,atempo={speed}"
        subprocess.run(["ffmpeg", "-i", r, "-af", filt, f, "-y"], capture_output=True)

    def process_video(self, f):
        vpath = os.path.join(self.video_folder, f)
        logging.info(f"=== PROCESSING FILE: {f} for languages: {','.join(self.target_langs)} ===")
        start_all = time.perf_counter(); ddir = self._cleanup_debug(f); seg_dir = os.path.join(ddir, "segments") if self.debug_mode else None
        if seg_dir: os.makedirs(seg_dir, exist_ok=True)
        monitor_state = {}
        self.monitor = ResourceMonitor(monitor_state); self.monitor.start()
        if not self.llm: threading.Thread(target=self._load_model).start()
        
        def step(name, func, *args):
            logging.info(f"STARTING STEP: {name}"); t = time.perf_counter(); res = func(*args)
            self.durations[name] = time.perf_counter() - t
            logging.info(f"COMPLETED STEP: {name} in {humanfriendly.format_timespan(self.durations[name])}")
            return res
        
        a_stereo, vocals = step("1. Audio Separation (Demucs)", audio_processor.prep_audio, vpath)
        diar, trans, audio_durs = step("2. Audio Analysis (Whisper/Diarization)", audio_processor.analyze_audio, vocals, self.gpu_audio)
        self.durations.update(audio_durs)
        script = self._create_script(diar, trans)
        if self.debug_mode:
            with open(os.path.join(ddir, "script_initial.json"), "w") as f_dbg: json.dump(script, f_dbg, indent=2)
        if not self.llm_ready.is_set(): logging.info("Waiting for LLM..."); self.llm_ready.wait()
        if self.abort_event.is_set(): return
        ref_subs = self._extract_subtitles(vpath)
        step("3. LLM Enhancement (Context/Speakers/ASR Fix)", self._llm_phase, script, ddir, ref_subs)
        step("4. Voice Reference Extraction", self._extract_refs, script, vocals, ddir)
        
        for lang in self.target_langs:
            logging.info(f"--- STARTING PRODUCTION FOR LANGUAGE: {lang} ---")
            q_text = queue.Queue(maxsize=15); q_audio = queue.Queue(maxsize=10)
            monitor_state['q_text'] = q_text; monitor_state['q_audio'] = q_audio
            res = []; draft_translations = []; final_translations = []
            p_th = threading.Thread(target=self._producer, args=(script, lang, q_text, self.speaker_info, self.global_context, draft_translations, final_translations))
            tts_th = threading.Thread(target=self._tts_worker, args=(lang, q_text, q_audio, []))
            post_th = threading.Thread(target=self._audio_postprocessor, args=(q_audio, res))
            p_th.start(); tts_th.start(); post_th.start()
            while p_th.is_alive() or tts_th.is_alive() or post_th.is_alive():
                if self.abort_event.is_set(): sys.exit(1)
                time.sleep(1.0)
            p_th.join(); tts_th.join(); post_th.join()
            if self.debug_mode:
                with open(os.path.join(ddir, f"translations_draft_{lang}.json"), "w") as f_out: json.dump(draft_translations, f_out, indent=2, ensure_ascii=False)
                with open(os.path.join(ddir, f"translations_final_{lang}.json"), "w") as f_out: json.dump(final_translations, f_out, indent=2, ensure_ascii=False)
                for path, _, _ in res: shutil.copy(path, os.path.join(seg_dir, f"{lang}_{os.path.basename(path)}"))
            res.sort(key=lambda x: x[1]); final_a = os.path.join(self.temp_dir, f"final_{lang}.ac3")
            step(f"6. Final Mix ({lang})", audio_processor.mix_audio, a_stereo, res, final_a)
            step(f"7. Muxing ({lang})", audio_processor.mux_video, vpath, final_a, lang, os.path.join(self.output_folder, f"dub_{lang}_{f}"), LANG_MAP.get(lang, lang))
            
        if self.monitor: self.monitor.stop()
        if self.llm: del self.llm; gc.collect(); torch.cuda.empty_cache()
        self._print_report(f, time.perf_counter() - start_all)

    def _print_report(self, f, t):
        avg_tps = self.llm_stats["tokens"] / self.llm_stats["time"] if self.llm_stats["time"] > 0 else 0
        rep = ["\n" + "="*50, f" PROFILING REPORT: {f}", "="*50]
        for k, v in sorted(self.durations.items()): rep.append(f" - {k:35} : {humanfriendly.format_timespan(v)}")
        rep.extend(["-"*50, f" - LLM PERFORMANCE                  : {avg_tps:.2f} tokens/s", "-"*50, f" TOTAL PROCESSING TIME: {humanfriendly.format_timespan(t)}", "="*50 + "\n"])
        logging.info("\n".join(rep))

    def run(self):
        try:
            for f in os.listdir(self.video_folder):
                if f.endswith((".mkv", ".mp4")): self.process_video(f)
        except Exception: logging.exception("FATAL ERROR in main loop")

if __name__ == "__main__":
    try: AIDubber().run()
    except Exception:
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()]); logging.exception("FATAL ERROR")
        raise