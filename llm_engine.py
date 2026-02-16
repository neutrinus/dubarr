import os
import json
import logging
import time
import threading
import gc
import torch
from typing import List, Dict, Optional
from llama_cpp import Llama, llama_supports_gpu_offload
from utils import parse_json, clean_output, count_syllables
from prompts import T_ANALYSIS, T_ED, T_TRANS_SYSTEM, T_CRITIC_SYSTEM, T_TRANS, T_CRITIC, T_SHORTEN
from config import LANG_MAP


class LLMManager:
    def __init__(
        self,
        model_path: str,
        device: str,
        inference_lock: Optional[threading.Lock],
        debug_mode: bool = False,
        target_langs: List[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.inference_lock = inference_lock
        self.debug_mode = debug_mode
        self.target_langs = target_langs or ["pl"]
        self.llm = None
        self.llm_stats = {"tokens": 0, "time": 0}
        self.ready_event = threading.Event()
        self.abort_event = threading.Event()

    def load_model(self):
        """Loads the LLM into VRAM or RAM. Downloads if missing. Skips in MOCK_MODE."""
        if os.environ.get("MOCK_MODE") == "1":
            logging.info("LLM: MOCK_MODE enabled. Skipping model load.")
            self.ready_event.set()
            return

        if not os.path.exists(self.model_path):
            logging.info(f"LLM: Model not found at {self.model_path}. Starting automatic download...")
            try:
                from huggingface_hub import hf_hub_download

                repo_id = "bartowski/google_gemma-3-12b-it-GGUF"
                filename = os.path.basename(self.model_path)

                # Ensure models directory exists
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=os.path.dirname(self.model_path),
                    local_dir_use_symlinks=False,
                )
                logging.info("LLM: Download completed successfully.")
            except Exception as e:
                logging.error(f"LLM: Failed to download model: {e}")
                self.abort_event.set()
                raise

        logging.info(f"LLM: Loading on {self.device}...")
        try:
            logging.info(f"LLM: GPU Offload Supported: {llama_supports_gpu_offload()}")

            # Determine params based on device type
            if "cuda" in self.device:
                n_gpu_layers = 99
                main_gpu = int(self.device.split(":")[-1])
            else:
                n_gpu_layers = 0  # CPU mode
                main_gpu = 0

            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                main_gpu=main_gpu,
                n_ctx=8192,
                n_batch=512,
                n_threads=4,
                flash_attn=(n_gpu_layers > 0),  # Flash Attn only works with CUDA
                verbose=self.debug_mode,
            )
            logging.info("LLM: Ready.")
            self.ready_event.set()
        except Exception as e:
            logging.error(f"LLM: Load Failed: {e}")
            self.abort_event.set()
            raise

    def _run_inference(self, prompt, **kwargs):
        """Wrapper for LLM inference that respects the global lock if needed."""
        if os.environ.get("MOCK_MODE") == "1":
            # Mock responses based on prompt type
            if "Analyze the movie script" in prompt:
                return {"choices": [{"text": json.dumps({
                    "summary": "Mock summary",
                    "glossary": {"test": "test"},
                    "speakers": {"SPEAKER_00": {"name": "Mock", "desc": "Male voice"}}
                })}]}
            if "Transcription Corrector" in prompt:
                return {"choices": [{"text": json.dumps({"0": "Mocked source correction"})}]}
            if "adapt" in prompt:
                return {"choices": [{"text": json.dumps({
                    "thought": "Mock thought",
                    "translations": [{"id": 0, "text": "To jest mockowe tłumaczenie"}]
                })}]}
            return {"choices": [{"text": "{}"}]}

        if not self.llm:
            raise RuntimeError("LLM model not loaded!")

        if self.inference_lock:
            with self.inference_lock:
                return self.llm(prompt, **kwargs)
        else:
            return self.llm(prompt, **kwargs)

    def analyze_script(self, script: List[Dict], ddir: str, subtitles: str = "") -> Dict:
        """Phase 1: Global Analysis & Phase 2: Speaker Profiling & Phase 3: ASR Correction."""

        # 1. Overview for analysis
        max_overview_lines = 400
        if len(script) > max_overview_lines:
            logging.warning(f"Script too long ({len(script)} lines). Truncating overview for Profiler.")
            mid = len(script) // 2
            ov_subset = script[mid - 200 : mid + 200]
            ov = "\n".join([f"{s['speaker']}: {s['text_en']}" for s in ov_subset])
        else:
            ov = "\n".join([f"{s['speaker']}: {s['text_en']}" for s in script])

        lang_names = [LANG_MAP.get(lang, lang) for lang in self.target_langs]
        logging.info("LLM: Starting Full Analysis (Context + Profiling)")

        t0 = time.perf_counter()
        res = self._run_inference(
            T_ANALYSIS.format(overview=ov, langs=", ".join(lang_names), subtitles=subtitles),
            max_tokens=2000,
            stop=["<|im_end|>"],
        )
        self._update_stats(res, time.perf_counter() - t0)

        analysis = parse_json(res["choices"][0]["text"])
        global_context = {k: v for k, v in analysis.items() if k != "speakers"}
        speaker_info = analysis.get("speakers", {})

        # 2. ASR Correction
        logging.info("LLM: Starting ASR Correction (Editor)")
        chunk_sz = 100
        glossary_str = json.dumps(global_context.get("glossary", {}))
        for i in range(0, len(script), chunk_sz):
            chunk = script[i : i + chunk_sz]
            txt = "\n".join([f"L_{i + j}: {s['text_en']}" for j, s in enumerate(chunk)])
            t0 = time.perf_counter()
            res = self._run_inference(
                T_ED.format(glossary=glossary_str, subtitles=subtitles, txt=txt),
                max_tokens=2000,
                temperature=0.0,
                stop=["<|im_end|>"],
            )
            self._update_stats(res, time.perf_counter() - t0)

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
                logging.warning(f"Failed to parse ASR corrections for chunk {i}: {e}")

        if self.debug_mode:
            self._dump_debug(ddir, global_context, speaker_info, script)

        return {"context": global_context, "speakers": speaker_info}

    def translation_producer(
        self,
        script: List[Dict],
        lang: str,
        q_text: threading.Thread,
        speaker_info: Dict,
        global_context: Dict,
        draft_list: List,
        final_list: List,
    ):
        """Threaded producer for translations and critiques."""
        t_trans_total = 0
        t_critic_total = 0
        try:
            summary = global_context.get("summary", "")
            glossary = global_context.get("glossary", {})
            lang_name = LANG_MAP.get(lang, lang)
            valid_indices = [
                idx
                for idx, s in enumerate(script)
                if not any(x in s["text_en"].lower() for x in ["[", "(", "laughter", "screams"])
            ]

            for b_start in range(0, len(valid_indices), 10):
                if self.abort_event.is_set():
                    return
                batch_indices = valid_indices[b_start : b_start + 10]
                json_batch = [
                    {
                        "id": idx,
                        "speaker": speaker_info.get(script[idx]["speaker"], {}).get("name", script[idx]["speaker"]),
                        "text": script[idx]["text_en"],
                    }
                    for idx in batch_indices
                ]

                # 1. Translation
                trans_prompt = T_TRANS.format(
                    system_prompt=T_TRANS_SYSTEM.format(
                        lang_name=lang_name, glossary=json.dumps(glossary, ensure_ascii=False)
                    ),
                    json_input=json.dumps(json_batch, indent=2),
                )
                t0 = time.perf_counter()
                res = self._run_inference(trans_prompt, max_tokens=1500, temperature=0.0, stop=["<|im_end|>"])
                dt = time.perf_counter() - t0
                t_trans_total += dt
                self._update_stats(res, dt)

                parsed_trans = parse_json(res["choices"][0]["text"])
                trans_map = {
                    item["id"]: clean_output(item["text"], self.target_langs)
                    for item in parsed_trans.get("translations", [])
                    if "id" in item
                }

                # 2. Critique
                json_critic = []
                final_batch_map = {}
                for idx in batch_indices:
                    txt = trans_map.get(idx, script[idx]["text_en"])
                    draft_list.append(
                        {
                            "index": idx,
                            "text": txt,
                            "speaker": script[idx]["speaker"],
                            "start": script[idx]["start"],
                            "end": script[idx]["end"],
                        }
                    )
                    if count_syllables(txt, lang) <= 3:
                        final_batch_map[idx] = txt
                    else:
                        json_critic.append({"id": idx, "original": script[idx]["text_en"], "draft": txt})

                if json_critic:
                    critic_prompt = T_CRITIC.format(
                        system_prompt=T_CRITIC_SYSTEM.format(
                            summary=summary, glossary=json.dumps(glossary, ensure_ascii=False)
                        ),
                        json_input=json.dumps(json_critic, indent=2),
                    )
                    t0 = time.perf_counter()
                    res_crit = self._run_inference(critic_prompt, max_tokens=1500, temperature=0.0, stop=["<|im_end|>"])
                    dt = time.perf_counter() - t0
                    t_critic_total += dt
                    self._update_stats(res_crit, dt)

                    parsed_crit = parse_json(res_crit["choices"][0]["text"])
                    for item in parsed_crit.get("final_translations", []):
                        if "id" in item:
                            final_batch_map[item["id"]] = clean_output(
                                item.get("final_text"), self.target_langs
                            ) or trans_map.get(item["id"])

                # 3. Finalization (Shortening if needed)
                for idx in batch_indices:
                    txt = final_batch_map.get(idx, trans_map.get(idx, script[idx]["text_en"]))
                    syl_orig = count_syllables(script[idx]["text_en"], "en")
                    syl_final = count_syllables(txt, lang)
                    if syl_final > syl_orig * 1.5:
                        t0 = time.perf_counter()
                        short_res = self._run_inference(
                            T_SHORTEN.format(original=script[idx]["text_en"], text=txt),
                            max_tokens=150,
                            temperature=0.0,
                            stop=["<|im_end|>"],
                        )
                        self._update_stats(short_res, time.perf_counter() - t0)
                        short_txt = clean_output(
                            parse_json(short_res["choices"][0]["text"]).get("final_text", txt), self.target_langs
                        )
                        if len(short_txt) > 2:
                            txt = short_txt

                    final_item = {
                        "index": idx,
                        "text": txt,
                        "speaker": script[idx]["speaker"],
                        "start": script[idx]["start"],
                        "end": script[idx]["end"],
                    }
                    final_list.append(final_item)
                    q_text.put(final_item)

            return {"t_trans": t_trans_total, "t_critic": t_critic_total}
        except Exception:
            logging.exception("LLM: Producer thread failed")
            self.abort_event.set()
        finally:
            q_text.put(None)

    def _update_stats(self, res, dt):
        toks = res.get("usage", {}).get("completion_tokens", 0)
        self.llm_stats["tokens"] += toks
        self.llm_stats["time"] += dt

    def _dump_debug(self, ddir, context, speakers, script):
        with open(os.path.join(ddir, "context.json"), "w") as f:
            json.dump(context, f, indent=2)
        with open(os.path.join(ddir, "speakers.json"), "w") as f:
            json.dump(speakers, f, indent=2)
        with open(os.path.join(ddir, "script_fixed.json"), "w") as f:
            json.dump(script, f, indent=2)

    def __del__(self):
        if self.llm:
            del self.llm
            gc.collect()
            if "cuda" in self.device:
                torch.cuda.empty_cache()
