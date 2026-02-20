import os
import json
import logging
import time
import threading
import gc
from typing import List, Dict, Optional
from utils import parse_json, clean_output
from prompts import T_ANALYSIS, T_ED, T_TRANS_SYSTEM, T_TRANS, T_REFINE_DURATION
from config import LANG_MAP, MOCK_MODE
from core.gpu_manager import GPUManager

try:
    import torch
except ImportError:
    torch = None

try:
    from llama_cpp import Llama, llama_supports_gpu_offload
except (ImportError, RuntimeError, OSError):
    logging.warning("LLM: Failed to import llama_cpp (likely missing CUDA libs). LLM will not be available.")
    Llama = None

    def llama_supports_gpu_offload():
        return False


class LLMManager:
    def __init__(
        self,
        model_path: str,
        inference_lock: Optional[threading.Lock],
        debug_mode: bool = False,
        target_langs: List[str] = None,
        abort_event: Optional[threading.Event] = None,
    ):
        self.model_path = model_path
        self.device = "cpu" # Default
        self.inference_lock = inference_lock
        self.debug_mode = debug_mode
        self.target_langs = target_langs or ["pl"]
        self.llm = None
        self.llm_stats = {"tokens": 0, "time": 0}
        self.ready_event = threading.Event()
        self.abort_event = abort_event or threading.Event()
        self.status = "IDLE"  # IDLE, LOADING, READY, ERROR

    def load_model(self):
        """Loads the LLM into VRAM or RAM. Downloads if missing. Skips in MOCK_MODE."""
        self.status = "LOADING"
        if MOCK_MODE:
            logging.info("LLM: MOCK_MODE enabled. Skipping model load.")
            self.status = "READY"
            self.ready_event.set()
            return

        if Llama is None:
            logging.error("LLM: llama_cpp library is not available.")
            self.status = "ERROR"
            self.abort_event.set()
            raise RuntimeError("llama-cpp-python import failed. Check logs for details (missing libcuda?).")

        try:
            # Dynamic GPU allocation
            # Gemma 12B Q4 needs approx 9GB total (weights + context + overhead)
            self.device = GPUManager.get_best_gpu(needed_mb=9000, purpose="LLM (Gemma)")

            if not os.path.exists(self.model_path):
                logging.info(f"LLM: Model not found at {self.model_path}. Starting automatic download...")
                self.status = "DOWNLOADING"
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
                self.status = "LOADING"

            logging.info(f"LLM: Loading on {self.device}...")
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
                n_ctx=4096,
                n_batch=512,
                n_threads=4,
                flash_attn=(n_gpu_layers > 0),  # Flash Attn only works with CUDA
                verbose=False,
            )
            logging.info("LLM: Ready.")
            self.status = "READY"
        except Exception as e:
            logging.error(f"LLM: Load Failed: {e}")
            self.status = "ERROR"
            self.abort_event.set()
        finally:
            self.ready_event.set()

    def _run_inference(self, prompt, **kwargs):
        """Wrapper for LLM inference that respects the global lock if needed."""
        if MOCK_MODE:
            # Mock responses based on prompt type
            if "Analyze the movie script" in prompt:
                return {
                    "choices": [
                        {
                            "text": json.dumps(
                                {
                                    "summary": "Mock summary",
                                    "glossary": {"test": "test"},
                                    "speakers": {"SPEAKER_00": {"name": "Mock", "desc": "Male voice"}},
                                }
                            )
                        }
                    ]
                }
            if "Transcription Corrector" in prompt:
                return {"choices": [{"text": json.dumps({"0": "Mocked source correction"})}]}
            if "adapt" in prompt:
                return {
                    "choices": [
                        {
                            "text": json.dumps(
                                {"thought": "Mock thought", "translations": [{"id": 0, "text": "To jest mockowe tłumaczenie"}]}
                            )
                        }
                    ]
                }
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

    def generate_drafts(
        self,
        script: List[Dict],
        lang: str,
        speaker_info: Dict,
        global_context: Dict,
    ) -> List[Dict]:
        """Generates initial draft translations for the entire script (batched)."""
        drafts = []
        glossary = global_context.get("glossary", {})
        lang_name = LANG_MAP.get(lang, lang)

        valid_indices = [
            idx
            for idx, s in enumerate(script)
            if not any(x in s["text_en"].lower() for x in ["[", "(", "laughter", "screams"])
        ]

        # Batch processing
        batch_size = 10
        for b_start in range(0, len(valid_indices), batch_size):
            if self.abort_event.is_set():
                break

            batch_indices = valid_indices[b_start : b_start + batch_size]
            json_batch = [
                {
                    "id": idx,
                    "speaker": speaker_info.get(script[idx]["speaker"], {}).get("name", script[idx]["speaker"]),
                    "text": script[idx]["text_en"],
                    "duration_sec": round(script[idx]["end"] - script[idx]["start"], 2),
                }
                for idx in batch_indices
            ]

            trans_prompt = T_TRANS.format(
                system_prompt=T_TRANS_SYSTEM.format(lang_name=lang_name, glossary=json.dumps(glossary, ensure_ascii=False)),
                json_input=json.dumps(json_batch, indent=2),
            )

            t0 = time.perf_counter()
            try:
                res = self._run_inference(trans_prompt, max_tokens=2000, temperature=0.1, stop=["<|im_end|>"])
                self._update_stats(res, time.perf_counter() - t0)

                parsed_trans = parse_json(res["choices"][0]["text"])
                trans_map = {
                    item["id"]: clean_output(item["text"], self.target_langs)
                    for item in parsed_trans.get("translations", [])
                    if "id" in item
                }

                for idx in batch_indices:
                    txt = trans_map.get(idx, script[idx]["text_en"])  # Fallback to original
                    drafts.append(
                        {
                            "index": idx,
                            "text": txt,
                            "speaker": script[idx]["speaker"],
                            "start": script[idx]["start"],
                            "end": script[idx]["end"],
                            "text_en": script[idx]["text_en"],  # Keep original for reference
                        }
                    )

            except Exception as e:
                logging.error(f"LLM: Draft generation failed for batch {b_start}: {e}")
                # Fallback for failed batch
                for idx in batch_indices:
                    drafts.append(
                        {
                            "index": idx,
                            "text": script[idx]["text_en"],  # Use original as fallback
                            "speaker": script[idx]["speaker"],
                            "start": script[idx]["start"],
                            "end": script[idx]["end"],
                            "text_en": script[idx]["text_en"],
                        }
                    )

        return drafts

    def refine_translation_by_duration(
        self,
        original_text: str,
        current_text: str,
        actual_dur: float,
        target_dur: float,
        glossary: Dict,
    ) -> str:
        """Refines text based on actual audio duration feedback."""
        delta = actual_dur - target_dur
        status = "TOO LONG" if delta > 0 else "TOO SHORT"

        # Don't refine if diff is negligible (< 0.5s or < 10%)
        if abs(delta) < 0.5 and abs(delta / (target_dur + 0.1)) < 0.1:
            return current_text

        logging.info(f"LLM: Refining '{current_text}' ({status}: {actual_dur:.2f}s vs {target_dur:.2f}s)")

        prompt = T_REFINE_DURATION.format(
            original_text=original_text,
            current_text=current_text,
            actual_duration=round(actual_dur, 2),
            target_duration=round(target_dur, 2),
            status=status,
            delta=round(delta, 2),
            glossary=json.dumps(glossary, ensure_ascii=False),
        )

        t0 = time.perf_counter()
        res = self._run_inference(prompt, max_tokens=200, temperature=0.7, stop=["<|im_end|>"])  # Higher temp for creativity
        self._update_stats(res, time.perf_counter() - t0)

        try:
            parsed = parse_json(res["choices"][0]["text"])
            final_text = clean_output(parsed.get("final_text", current_text), self.target_langs)
            if final_text and len(final_text) > 1:
                logging.info(f"LLM: Refined -> '{final_text}'")
                return final_text
        except Exception as e:
            logging.warning(f"LLM: Failed to parse refinement: {e}")

        return current_text

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
            if torch and "cuda" in self.device:
                torch.cuda.empty_cache()
