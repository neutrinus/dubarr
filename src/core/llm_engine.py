import os
import json
import logging
import time
import threading
import gc
import re
from typing import List, Dict, Optional, Any
from utils import (
    translation_glossary,
    parse_json,
    clean_output,
    count_syllables,
    respelling_heard_as,
    clean_arpabet,
    check_polish_spelling,
    strip_title,
)
from prompts import (
    T_ANALYSIS,
    T_ED,
    T_TRANS_SYSTEM,
    T_REVIEW,
    T_REFINE_DURATION,
    T_GLOSSARY_READ,
    T_GLOSSARY_RESPELL,
    T_GLOSSARY_CMU,
    T_TERMS,
    RESPELL_RULES,
)
from config import (
    LANG_MAP,
    MOCK_MODE,
    LLM_MODEL_REPO,
    LLM_API_URL,
    LLM_API_MODEL,
    LLM_API_KEY,
    LLM_TIMEOUT,
    LLM_N_CTX,
    LLM_THREADS,
    LLM_MAX_TOKENS,
    DUB_SYLLABLES_PER_SECOND,
    TRANSLATION_WINDOW,
    GLOSSARY_MODE,
)
from core.gpu_manager import GPUManager
from infrastructure.llm_client import OpenAICompatibleClient

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


def speaker_name(speaker_info: Dict, speaker_id: str) -> str:
    info = speaker_info.get(speaker_id)
    return info.get("name") or speaker_id if isinstance(info, dict) else speaker_id


def speaker_gender(speaker_info: Dict, speaker_id: str) -> str:
    """'male'/'female' from the analysis (explicit field, else guessed from the free-text profile), or ''."""
    info = speaker_info.get(speaker_id)
    if not isinstance(info, dict):
        return ""
    gender = str(info.get("gender", "")).strip().lower()
    if gender in ("male", "female"):
        return gender
    desc = str(info.get("desc", "")).lower()
    if re.search(r"\b(female|woman|girl|lady)\b", desc):
        return "female"
    if re.search(r"\b(male|man|boy|guy)\b", desc):
        return "male"
    return ""


def _terms_text(global_context: Dict, lang: str) -> str:
    terms = (global_context.get("terms") or {}).get(lang) or {}
    return json.dumps(terms, ensure_ascii=False) if terms else "(none)"


def _address_text(global_context: Dict, lang: str) -> str:
    address = (global_context.get("address") or {}).get(lang) or {}
    return "\n".join(f"- {k}: {v}" for k, v in sorted(address.items())) if address else "(none)"


def _found_in(term: str, text: str) -> bool:
    """Whether every word of ``term`` occurs in ``text``, inflected (Polish endings change: "Gwiezdnej Floty")."""
    words = re.findall(r"\w+", term.lower())
    return bool(words) and all(w[: max(3, len(w) - 3)] in text for w in words)


def apply_editor_flags(script: List[Dict], flags: Dict, doubts: Dict, fixes: Optional[Dict] = None):
    """Marks each line's ``asr`` (sure / unsure / fixed / guessed / hallucination) and ``speaker_doubt`` (Spec 001,
    2.2c step 4). Also used to re-apply a cached Stage 3."""
    fixes = fixes or {}
    for idx, line in enumerate(script):
        flag = flags.get(str(idx))
        if flag in ("guessed", "hallucination"):
            line["asr"] = flag
        elif line.get("asr_unsure"):
            line["asr"] = "fixed" if str(idx) in fixes else "unsure"
        else:
            line["asr"] = "fixed" if str(idx) in fixes else "sure"
        if str(idx) in doubts:
            line["speaker_doubt"] = doubts[str(idx)]


def log_editor_flags(script: List[Dict], doubts: Dict):
    counts: Dict[str, int] = {}
    for line in script:
        counts[line.get("asr", "sure")] = counts.get(line.get("asr", "sure"), 0) + 1
    logging.info("ASR check: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    by_speaker: Dict[str, List[str]] = {}
    for idx, reason in doubts.items():
        if 0 <= int(idx) < len(script):
            by_speaker.setdefault(script[int(idx)].get("speaker", "?"), []).append(f"L_{idx}: {reason}")
    for spk, reasons in sorted(by_speaker.items(), key=lambda x: -len(x[1])):
        logging.warning(f"Speaker doubt: {spk} ({len(reasons)} lines) - " + " | ".join(reasons[:3]))


def speaker_label(speaker_info: Dict, speaker_id: str) -> str:
    """'Name (female)' for the per-line prompts."""
    gender = speaker_gender(speaker_info, speaker_id)
    return speaker_name(speaker_info, speaker_id) + (f" ({gender})" if gender else "")


def _untranslated(text: str, source: str) -> bool:
    """A "translation" that repeats a source line of at least three words: the model answered in English."""

    def words(x: str) -> List[str]:
        return "".join(ch for ch in x.lower() if ch.isalnum() or ch == " ").split()

    return len(words(source)) >= 3 and words(text) == words(source)


class LLMManager:
    def __init__(
        self,
        model_path: str,
        inference_lock: Optional[threading.Lock],
        debug_mode: bool = False,
        target_langs: List[str] = None,
        abort_event: Optional[threading.Event] = None,
        service: Optional[Any] = None,
    ):
        self.model_path = model_path
        self.device = "cpu"  # Default
        self.inference_lock = inference_lock
        self.debug_mode = debug_mode
        self.target_langs = target_langs or ["pl"]
        self.llm = None
        self.service = service
        self.llm_stats = {"tokens": 0, "time": 0}
        self.ready_event = threading.Event()
        self.abort_event = abort_event or threading.Event()
        self._load_lock = threading.Lock()
        self._inference_thread_lock = threading.Lock()
        self.status = "IDLE"  # IDLE, LOADING, READY, ERROR
        # With LLM_API_URL set, inference goes to an external server and nothing is loaded here
        self.client = (
            OpenAICompatibleClient(LLM_API_URL, LLM_API_MODEL, LLM_API_KEY, LLM_TIMEOUT)
            if LLM_API_URL and not MOCK_MODE
            else None
        )

    def ensure_model_downloaded(self):
        """Ensures the model file exists locally. Downloads if missing. Does NOT load to VRAM."""
        if MOCK_MODE or self.client:
            return

        if not os.path.exists(self.model_path):
            logging.info(f"LLM: Model not found at {self.model_path}. Starting automatic download...")
            from huggingface_hub import hf_hub_download

            repo_id = LLM_MODEL_REPO
            filename = os.path.basename(self.model_path)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(self.model_path),
                local_dir_use_symlinks=False,
            )
            logging.info("LLM: Download completed successfully.")

    def load_model(self):
        """Loads the LLM into VRAM or RAM. Skips in MOCK_MODE. Thread-safe."""
        if self.status == "READY" or self.llm is not None:
            return

        with self._load_lock:
            # Double-check after acquiring lock
            if self.status == "READY" or self.llm is not None:
                return

            self.status = "LOADING"
        if MOCK_MODE:
            logging.info("LLM: MOCK_MODE enabled. Skipping model load.")
            self.status = "READY"
            self.ready_event.set()
            return

        if self.client:
            logging.info(f"LLM: Using external server at {LLM_API_URL}")
            self.status = "READY"
            self.ready_event.set()
            return

        if Llama is None:
            logging.error("LLM: llama_cpp library is not available.")
            self.status = "ERROR"
            self.abort_event.set()
            raise RuntimeError("llama-cpp-python import failed. Check logs for details (missing libcuda?).")

        try:
            self.ensure_model_downloaded()

            # Gemma 12B Q4 + 8k context needs approx 10GB total
            self.device = GPUManager.get_best_gpu(needed_mb=10000, purpose="LLM (Gemma 3 12B)")

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
                n_ctx=LLM_N_CTX,
                n_batch=512,
                n_threads=LLM_THREADS,
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

    def shutdown(self):
        """Explicitly releases the LLM from VRAM."""
        if self.llm:
            logging.info("LLM: Unloading model...")
            del self.llm
            self.llm = None
            self.status = "IDLE"
            gc.collect()
            if torch and "cuda" in self.device:
                torch.cuda.empty_cache()
            logging.info("LLM: Unloaded.")

    def _run_inference(self, messages, priority=10, **kwargs):
        """Runs a chat completion (in-process model or external server) and returns
        {"choices": [{"text": ...}], "usage": {...}}. Respects the global lock or uses the RPC service."""
        if MOCK_MODE:
            return self._mock_response("\n".join(m["content"] for m in messages))

        if not self.llm:
            self.load_model()

        if self.service:

            def _direct_call(m, **kw):
                with self._inference_thread_lock:
                    try:
                        return self._chat(m, **kw)
                    except Exception as e:
                        logging.error(f"LLM Internal Error during inference: {e}", exc_info=True)
                        raise

            future = self.service.submit(_direct_call, messages, priority=priority, **kwargs)
            return future.result()

        if self.inference_lock:
            with self.inference_lock:
                return self._chat(messages, **kwargs)
        return self._chat(messages, **kwargs)

    def _chat(self, messages, **kwargs):
        # The model's own chat template is applied (by llama.cpp here or by the server); JSON output is enforced
        kwargs.setdefault("response_format", {"type": "json_object"})
        retry_budget = None
        for attempt in (1, 2):
            if self.client:
                res = self.client.chat(messages, **kwargs)
            elif self.llm:
                res = self.llm.create_chat_completion(messages=messages, **kwargs)
            else:
                raise RuntimeError("LLM failed to load!")

            choice = res["choices"][0]
            text = choice["message"]["content"] or ""
            if choice.get("finish_reason") != "length" or attempt == 2:
                break

            # A truncated answer (a reasoning model can spend the whole budget on reasoning) is retried
            # once with a bigger budget instead of silently coming back empty.
            budget = kwargs.get("max_tokens") or LLM_MAX_TOKENS
            retry_budget = max(budget, retry_budget or 0) * 2
            logging.warning(f"LLM: answer truncated at {budget} tokens, retrying with {retry_budget}.")
            kwargs["max_tokens"] = retry_budget

        return {
            "choices": [{"text": text, "finish_reason": choice.get("finish_reason")}],
            "usage": res.get("usage") or {},
        }

    @staticmethod
    def _mock_response(prompt):
        """Canned responses by prompt type for MOCK_MODE."""
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

    def analyze_script(
        self,
        script: List[Dict],
        ddir: str,
        subtitles: str = "",
        media_info: str = "",
        fixed_glossary: Optional[Dict[str, str]] = None,
        names: Optional[List[str]] = None,
        fixed_phonemes: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Phase 1: Global Analysis & Phase 2: Speaker Profiling & Phase 3: ASR Correction.

        ``fixed_glossary`` holds the series' settled spellings (Spec 006, 3), ``names`` the names known to occur.
        """
        self.load_model()
        fixed_glossary = fixed_glossary or {}
        names_str = ", ".join(names or []) or "(none known)"

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
        # Global Analysis is Priority 2
        res = self._run_inference(
            [
                {
                    "role": "system",
                    "content": T_ANALYSIS.format(
                        langs=", ".join(lang_names),
                        subtitles=subtitles,
                        media_info=media_info or "(nothing known)",
                        names=names_str,
                        fixed_glossary=json.dumps(fixed_glossary, ensure_ascii=False) if fixed_glossary else "(none yet)",
                    ),
                },
                {"role": "user", "content": ov},
            ],
            priority=2,
            max_tokens=LLM_MAX_TOKENS,
        )
        self._update_stats(res, time.perf_counter() - t0)

        analysis = parse_json(res["choices"][0]["text"])
        global_context = {k: v for k, v in analysis.items() if k != "speakers"}
        speaker_info = analysis.get("speakers", {})
        glossary = global_context.get("glossary")
        glossary = glossary if isinstance(glossary, dict) else {}
        # Ranks are translated, not respelled: "Captain James Tiberius Kirk" became "Kapten Dzejms Tajberius Kark"
        glossary = dict(strip_title(str(t), str(sp)) for t, sp in glossary.items() if isinstance(sp, (str, int, float)))
        global_context["glossary"] = glossary
        if GLOSSARY_MODE == "phonemes":
            # The translation keeps the original names; the TTS gets their English phonemes
            names_only = {str(t): str(t) for t in list(glossary) + list(fixed_phonemes or {})}
            global_context["glossary"] = names_only
            global_context["phonemes"] = self.glossary_phonemes(list(names_only), fixed_phonemes)
        else:
            # Polish respellings are checked by spelling rules against the names' English phonemes (Spec 006, 3)
            name_phonemes = {}
            if self.target_langs and self.target_langs[0] == "pl":
                spellings = {
                    str(t): str(sp) for t, sp in {**glossary, **fixed_glossary}.items() if isinstance(sp, (str, int, float))
                }
                # A name left in its English spelling is read by Polish rules too ("Jamie", "Belleci"), so it is checked
                terms = [t for t, sp in spellings.items() if not (t.isupper() and len(t) <= 6)]
                name_phonemes = self.glossary_phonemes(terms, fixed_phonemes) if terms else {}
            replaced: set = set()
            global_context["glossary"] = self.check_glossary(
                glossary, fixed_glossary, phonemes=name_phonemes, replaced=replaced
            )
            global_context["name_phonemes"] = name_phonemes
            global_context["glossary_replaced"] = sorted(replaced)

        # 2. ASR Correction; unsure lines come with their other hearings (Spec 001, 2.2c step 4)
        logging.info("LLM: Starting ASR Correction (Editor)")
        chunk_sz = 100
        script_fixes: Dict[str, str] = {}
        script_flags: Dict[str, str] = {}
        speaker_doubts: Dict[str, str] = {}
        glossary_str = json.dumps(global_context.get("glossary", {}))
        summary = str(global_context.get("summary") or "")[:1500]
        for i in range(0, len(script), chunk_sz):
            chunk = script[i : i + chunk_sz]
            txt = "\n".join(self._editor_line(i + j, s, speaker_info) for j, s in enumerate(chunk))
            t0 = time.perf_counter()
            res = self._run_inference(
                [
                    {
                        "role": "system",
                        "content": T_ED.format(glossary=glossary_str, subtitles=subtitles, names=names_str, summary=summary),
                    },
                    {"role": "user", "content": txt},
                ],
                priority=2,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.0,
            )
            self._update_stats(res, time.perf_counter() - t0)
            try:
                fixes, flags, doubts = self._editor_answer(parse_json(res["choices"][0]["text"]), script, i, len(chunk))
            except Exception as e:
                logging.warning(f"Failed to parse ASR corrections for chunk {i}: {e}")
                continue
            for idx, new_text in fixes.items():
                if script[idx]["text_en"] != new_text and self.debug_mode:
                    logging.info(f"ASR Fix L_{idx}: '{script[idx]['text_en']}' -> '{new_text}'")
                script[idx]["text_en"] = new_text
                script_fixes[str(idx)] = new_text
            script_flags.update({str(k): v for k, v in flags.items()})
            speaker_doubts.update({str(k): v for k, v in doubts.items()})
        apply_editor_flags(script, script_flags, speaker_doubts, script_fixes)
        log_editor_flags(script, speaker_doubts)

        if self.debug_mode:
            self._dump_debug(ddir, global_context, speaker_info, script)

        # The corrections are returned too, so a resumed job can re-apply them from the step cache
        return {
            "context": global_context,
            "speakers": speaker_info,
            "script_fixes": script_fixes,
            "script_flags": script_flags,
            "speaker_doubts": speaker_doubts,
        }

    @staticmethod
    def _editor_line(idx: int, line: Dict, speaker_info: Dict) -> str:
        """`L_12 (Chris Rose): text`, and `|| UNSURE` with the other hearings for an unsure line."""
        out = f"L_{idx} ({speaker_label(speaker_info, line.get('speaker', ''))}): {line['text_en']}"
        if line.get("asr_unsure"):
            hearings = line.get("asr_hearings") or {}
            out += " || UNSURE" + "".join(f'; {kind}: "{text}"' for kind, text in sorted(hearings.items()))
        return out

    @staticmethod
    def _editor_answer(answer, script: List[Dict], start: int, size: int):
        """(fixes {idx: text}, flags {idx: guessed|hallucination}, doubts {idx: reason}) from the corrector's JSON.

        The old flat answer ({"12": "text"}) is read as fixes. Indices outside the chunk are ignored.
        """
        if not isinstance(answer, dict):
            return {}, {}, {}
        structured = any(k in answer for k in ("fixes", "guessed", "hallucinations", "speaker_doubts"))
        raw_fixes = answer.get("fixes") if structured else answer

        def index(key):
            try:
                idx = int(str(key).replace("L_", "").strip())
            except ValueError:
                return None
            return idx if start <= idx < start + size and idx < len(script) else None

        fixes = {}
        for key, text in (raw_fixes or {}).items() if isinstance(raw_fixes, dict) else []:
            idx = index(key)
            if idx is not None and isinstance(text, str) and text.strip():
                fixes[idx] = text.strip()
        flags = {}
        for key in answer.get("guessed") or [] if structured else []:
            idx = index(key)
            if idx is not None:
                flags[idx] = "guessed"
        for key in answer.get("hallucinations") or [] if structured else []:
            idx = index(key)
            # Only a short or unsure line may be a hallucination: a long, confident one is speech
            if idx is not None and (script[idx].get("asr_unsure") or len(script[idx]["text_en"].split()) <= 4):
                flags[idx] = "hallucination"
        doubts = {}
        raw_doubts = answer.get("speaker_doubts") if structured else None
        for key, reason in (raw_doubts or {}).items() if isinstance(raw_doubts, dict) else []:
            idx = index(key)
            if idx is not None and isinstance(reason, str) and reason.strip():
                doubts[idx] = reason.strip()[:200]
        return fixes, flags, doubts

    def _hear_spellings(self, spellings: List[str], lang_name: str) -> List[str]:
        """What each spelling sounds like in English spelling ("" when the model gave nothing)."""
        res = self._run_inference(
            [
                {"role": "system", "content": T_GLOSSARY_READ.format(lang_name=lang_name)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"spellings": [{"id": i, "spelling": sp} for i, sp in enumerate(spellings)]}, ensure_ascii=False
                    ),
                },
            ],
            priority=2,
            temperature=0.0,
        )
        heard = [""] * len(spellings)
        for entry in parse_json(res["choices"][0]["text"]).get("heard") or []:
            try:
                heard[int(entry["id"])] = str(entry.get("english") or "")
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        return heard

    def _respell(self, failed: List[Dict], lang_name: str) -> Dict[str, str]:
        res = self._run_inference(
            [
                {
                    "role": "system",
                    "content": T_GLOSSARY_RESPELL.format(lang_name=lang_name, rules=RESPELL_RULES.get(lang_name, "")),
                },
                {"role": "user", "content": json.dumps({"names": failed}, ensure_ascii=False)},
            ],
            priority=2,
            temperature=0.3,
        )
        out = {}
        for entry in parse_json(res["choices"][0]["text"]).get("spellings") or []:
            try:
                out[failed[int(entry["id"])]["name"]] = str(entry.get("spelling") or "").strip()
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        return out

    def check_glossary(
        self,
        glossary: Dict,
        fixed: Optional[Dict[str, str]] = None,
        rounds: int = 2,
        phonemes: Optional[Dict[str, str]] = None,
        replaced: Optional[set] = None,
    ) -> Dict[str, str]:
        """Every respelling must pass the reading check; settled ones too when they can be checked (Spec 006, 3).

        With the name's English phonemes (``phonemes``) and Polish as the target, the respelling is read by Polish
        spelling rules and must keep the consonants of the phonemes; a "w" where English has W becomes "ł" in place.
        Otherwise the model reads it back as English and that must keep the term's consonants ("Rej-or-bek" is heard
        as "Rayorbeck" and loses the "z" of "Razorback"). A failed respelling is respelled up to ``rounds`` times; a
        term that still fails keeps its original spelling, which the TTS reads as written. Settled spellings are
        only checked by rules: one that fails is replaced, and its term is added to ``replaced``.
        """
        fixed = fixed or {}
        phonemes = phonemes or {}
        replaced = replaced if replaced is not None else set()
        result = {str(k): str(v) for k, v in glossary.items() if isinstance(v, (str, int, float)) and str(v).strip()}
        result.update(fixed)  # a stored spelling cannot be changed by the analysis
        lang = self.target_langs[0] if self.target_langs else ""
        lang_name = LANG_MAP.get(lang, lang) or "target language"
        by_rules = lang == "pl"

        def checkable(t, sp):
            if t.isupper() and len(t.replace(".", "")) <= 6:
                return False
            # An unchanged spelling can only be checked by rules: read back as English it always sounds right
            return sp.strip() != t.strip() or by_rule(t)

        def by_rule(term):
            return by_rules and term in phonemes

        pending = {t: sp for t, sp in result.items() if checkable(t, sp) and (t not in fixed or by_rule(t))}
        if MOCK_MODE or not pending:
            return result
        for attempt in range(rounds + 1):
            failed = []
            heard_terms = [t for t in pending if not by_rule(t)]
            heard = (
                dict(zip(heard_terms, self._hear_spellings([pending[t] for t in heard_terms], lang_name)))
                if heard_terms
                else {}
            )
            for term in list(pending):
                spelling = pending[term]
                if by_rule(term):
                    ok, spelling, lost = check_polish_spelling(spelling, phonemes[term])
                    if spelling != pending[term]:
                        logging.info(f"Glossary: '{pending[term]}' -> '{spelling}' for '{term}' (a Polish w is read as v)")
                    sounds = f"read in {lang_name}: {lost}; the name is {phonemes[term]}"
                else:
                    sounds = heard.get(term, "")
                    ok = bool(sounds) and respelling_heard_as(term, sounds)
                if ok:
                    pending.pop(term)
                    if term in fixed and spelling != fixed[term]:
                        replaced.add(term)
                    result[term] = spelling
                    continue
                logging.info(f"Glossary: '{spelling}' for '{term}' sounds like '{sounds}', rejected")
                failed.append({"id": len(failed), "name": term, "spelling": spelling, "sounds_like": sounds})
            if not failed:
                break
            if attempt == rounds:
                for f in failed:
                    logging.info(f"Glossary: keeping the original spelling of '{f['name']}'")
                    result[f["name"]] = f["name"]
                    if f["name"] in fixed:
                        replaced.add(f["name"])
                break
            for term, spelling in self._respell(failed, lang_name).items():
                if spelling:
                    pending[term] = spelling
        return result

    def glossary_phonemes(self, names: List[str], fixed: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """English CMU phonemes per name (GLOSSARY_MODE=phonemes); stored ones stay, invalid answers are dropped."""
        out = {t: p for t, p in (fixed or {}).items() if clean_arpabet(p)}
        todo = [n for n in names if n not in out and not (n.isupper() and len(n.replace(".", "")) <= 6)]
        if MOCK_MODE or not todo:
            return out
        res = self._run_inference(
            [
                {"role": "system", "content": T_GLOSSARY_CMU},
                {
                    "role": "user",
                    "content": json.dumps({"names": [{"id": i, "name": n} for i, n in enumerate(todo)]}, ensure_ascii=False),
                },
            ],
            priority=2,
            temperature=0.0,
        )
        for entry in parse_json(res["choices"][0]["text"]).get("phonemes") or []:
            try:
                name = todo[int(entry["id"])]
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            arpabet = clean_arpabet(str(entry.get("arpabet") or ""))
            if arpabet:
                out[name] = arpabet
            else:
                logging.info(f"Glossary: no valid phonemes for '{name}' ('{entry.get('arpabet')}'), it stays as written")
        return out

    @staticmethod
    def syllable_budget(line: Dict) -> Dict:
        """Spoken-syllable budget of a line from its time budget (Spec 001, 2.5a)."""
        speech = max(0.0, line["end"] - line["start"])
        limit = max(speech, line.get("max_end", line["end"]) - line.get("min_start", line["start"]))
        target = max(1, round(speech * DUB_SYLLABLES_PER_SECOND))
        return {
            "min_syllables": max(1, round(target * 0.75)),
            "target_syllables": target,
            "max_syllables": max(target, round(limit * DUB_SYLLABLES_PER_SECOND)),
        }

    @staticmethod
    def _is_sound_marker(text: str) -> bool:
        return any(x in text.lower() for x in ["[", "(", "laughter", "screams"])

    def _about(self, global_context: Dict) -> str:
        return (
            "\n".join(x for x in (global_context.get("media", ""), str(global_context.get("summary", "") or "")) if x)
            or "(unknown)"
        )

    def _draft_entry(self, script, idx, text, speaker_info, lang=""):
        return {
            "index": idx,
            "text": text,
            "speaker": script[idx]["speaker"],
            "start": script[idx]["start"],
            "end": script[idx]["end"],
            "max_end": script[idx].get("max_end", script[idx]["end"]),
            "min_start": script[idx].get("min_start", script[idx]["start"]),
            "text_en": script[idx]["text_en"],  # Keep original for reference
            "speaker_label": speaker_label(speaker_info, script[idx]["speaker"]),
            "subtitle": (script[idx].get("subtitles") or {}).get(lang, ""),
        }

    def generate_drafts(
        self,
        script: List[Dict],
        lang: str,
        speaker_info: Dict,
        global_context: Dict,
    ) -> List[Dict]:
        """Translates the script in windows with surrounding context and per-line syllable budgets (Spec 001, 2.5a)."""
        glossary = translation_glossary(global_context, lang)
        lang_name = LANG_MAP.get(lang, lang)
        system = T_TRANS_SYSTEM.format(
            lang_name=lang_name,
            context=self._about(global_context),
            glossary=json.dumps(glossary, ensure_ascii=False),
            terms=_terms_text(global_context, lang),
            address=_address_text(global_context, lang),
        )

        valid_indices = [idx for idx, s in enumerate(script) if not self._is_sound_marker(s["text_en"])]
        translated: Dict[int, str] = {}
        variants: Dict[int, List[str]] = {}

        position = {idx: n for n, idx in enumerate(valid_indices)}

        def translate(window: List[int]) -> None:
            first, last = position[window[0]], position[window[-1]]
            previous = valid_indices[max(0, first - 6) : first]
            following = valid_indices[last + 1 : last + 5]
            request = {
                "previous": [
                    {
                        "speaker": speaker_name(speaker_info, script[i]["speaker"]),
                        "text": script[i]["text_en"],
                        "translation": translated.get(i, ""),
                    }
                    for i in previous
                ],
                "translate": [],
                "next": [
                    {"speaker": speaker_name(speaker_info, script[i]["speaker"]), "text": script[i]["text_en"]}
                    for i in following
                ],
            }
            for idx in window:
                item = {
                    "id": idx,
                    "speaker": speaker_name(speaker_info, script[idx]["speaker"]),
                    "text": script[idx]["text_en"],
                }
                gender = speaker_gender(speaker_info, script[idx]["speaker"])
                if gender:
                    item["speaker_gender"] = gender
                item.update(self.syllable_budget(script[idx]))
                subtitle = (script[idx].get("subtitles") or {}).get(lang)
                if subtitle:
                    item["subtitle"] = subtitle
                request["translate"].append(item)

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(request, indent=2, ensure_ascii=False)},
            ]
            t0 = time.perf_counter()
            try:
                # Draft generation is Priority 2
                res = self._run_inference(messages, priority=2, max_tokens=LLM_MAX_TOKENS, temperature=0.1)
                self._update_stats(res, time.perf_counter() - t0)
                parsed = parse_json(res["choices"][0]["text"])
                for item in parsed.get("translations", []):
                    if "id" in item and item.get("text") and int(item["id"]) in window:
                        i = int(item["id"])
                        text = clean_output(item["text"], self.target_langs)
                        if _untranslated(text, script[i]["text_en"]):
                            continue
                        translated[i] = text
                        # Main translation first, then the length variants (Spec 001, 2.5a)
                        options = [translated[i]]
                        for key in ("short", "long"):
                            if isinstance(item.get(key), str) and item[key].strip():
                                option = clean_output(item[key], self.target_langs)
                                if option and option not in options:
                                    options.append(option)
                        variants[i] = options
            except Exception as e:
                logging.error(f"LLM: Draft generation failed for lines {window[0]}-{window[-1]}: {e}")

        for w_start in range(0, len(valid_indices), TRANSLATION_WINDOW):
            if self.abort_event.is_set():
                break
            window = valid_indices[w_start : w_start + TRANSLATION_WINDOW]
            translate(window)
            # Lines the answer left out (or left in English) are asked again, in halves (Spec 001, 2.5a)
            missing = [idx for idx in window if idx not in translated]
            if missing and not self.abort_event.is_set():
                logging.warning(f"LLM: No {lang_name} translation for lines {missing}, asking again")
                half = (len(missing) + 1) // 2
                for part in (missing[:half], missing[half:]):
                    if part:
                        translate(part)
                missing = [idx for idx in window if idx not in translated]
                if missing:
                    logging.warning(f"LLM: Still no {lang_name} translation for lines {missing}, keeping the original text")

        drafts = []
        for idx in valid_indices:
            entry = self._draft_entry(script, idx, translated.get(idx, script[idx]["text_en"]), speaker_info, lang)
            entry["variants"] = variants.get(idx, [entry["text"]])
            drafts.append(entry)
        return drafts

    def extract_terms(self, script: List[Dict], lang: str, speaker_info: Dict, chunk: int = 60) -> Dict:
        """{"terms": {source: target}, "address": {"A -> B": form}} that the episode's target-language subtitles show
        (Spec 006, 5). A pair is kept only when the source term is in the script and the target term in the subtitles."""
        lang_name = LANG_MAP.get(lang, lang)
        system = T_TERMS.format(lang_name=lang_name)
        lines = [
            {"speaker": speaker_name(speaker_info, s["speaker"]), "source": s["text_en"], "subtitle": s["subtitles"][lang]}
            for s in script
            if (s.get("subtitles") or {}).get(lang)
        ]
        source_text = " ".join(s.get("text_en", "") for s in script).lower()
        subtitle_text = " ".join(x["subtitle"] for x in lines).lower()
        terms: Dict[str, str] = {}
        address: Dict[str, str] = {}
        for start in range(0, len(lines), chunk):
            if self.abort_event.is_set():
                break
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"lines": lines[start : start + chunk]}, indent=1, ensure_ascii=False)},
            ]
            t0 = time.perf_counter()
            try:
                res = self._run_inference(messages, priority=2, max_tokens=LLM_MAX_TOKENS, temperature=0.1)
                self._update_stats(res, time.perf_counter() - t0)
                parsed = parse_json(res["choices"][0]["text"]) or {}
            except Exception as e:
                logging.warning(f"LLM: Term extraction failed for lines {start}+: {e}")
                continue
            for term, target in (parsed.get("terms") or {}).items():
                term, target = str(term).strip(), str(target or "").strip()
                if not term or not target or term.lower() == target.lower():
                    continue
                if term.lower() in source_text and _found_in(target, subtitle_text):
                    terms[term] = target
                else:
                    logging.debug(f"Terms ({lang}): dropping '{term}' -> '{target}' (not in the script and subtitles)")
            for entry in parsed.get("address") or []:
                if isinstance(entry, dict) and entry.get("speaker") and entry.get("addressee") and entry.get("form"):
                    address[f"{entry['speaker']} -> {entry['addressee']}"] = str(entry["form"]).strip()
        if terms or address:
            logging.info(f"Terms ({lang}) from the subtitles: {terms}; forms of address: {address}")
        return {"terms": terms, "address": address}

    def review_drafts(self, drafts: List[Dict], lang: str, speaker_info: Dict, global_context: Dict) -> List[Dict]:
        """Completeness review: a second pass fixes lines with missing or wrong information (Spec 001, 2.5a)."""
        lang_name = LANG_MAP.get(lang, lang)
        system = T_REVIEW.format(
            lang_name=lang_name,
            context=self._about(global_context),
            glossary=json.dumps(translation_glossary(global_context, lang), ensure_ascii=False),
            terms=_terms_text(global_context, lang),
            address=_address_text(global_context, lang),
        )
        by_index = {d["index"]: d for d in drafts}
        order = [d["index"] for d in drafts]

        for w_start in range(0, len(order), TRANSLATION_WINDOW):
            if self.abort_event.is_set():
                break
            window = order[w_start : w_start + TRANSLATION_WINDOW]
            lines = [
                {
                    "id": i,
                    "speaker": speaker_name(speaker_info, by_index[i]["speaker"]),
                    "source": by_index[i].get("text_en", ""),
                    "translation": by_index[i].get("text", ""),
                    "max_syllables": self.syllable_budget(by_index[i])["max_syllables"],
                    **({"subtitle": by_index[i]["subtitle"]} if by_index[i].get("subtitle") else {}),
                }
                for i in window
            ]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"lines": lines}, indent=2, ensure_ascii=False)},
            ]
            t0 = time.perf_counter()
            try:
                res = self._run_inference(messages, priority=2, max_tokens=LLM_MAX_TOKENS, temperature=0.1)
                self._update_stats(res, time.perf_counter() - t0)
                parsed = parse_json(res["choices"][0]["text"])
                for problem in parsed.get("problems", []) or []:
                    logging.info(f"Review ({lang}): {problem}")
                for fix in parsed.get("fixes", []) or []:
                    try:
                        i = int(fix.get("id"))
                    except (TypeError, ValueError):
                        continue
                    text = clean_output(fix.get("text") or "", self.target_langs)
                    if i not in window or not text:
                        continue
                    if _untranslated(text, by_index[i].get("text_en", "")):
                        continue
                    budget = self.syllable_budget(by_index[i])["max_syllables"]
                    # A fix far over budget would only be cut again in the sync loop; but a line the draft left in
                    # the original language takes any translation: the sync loop shortens it, English is dropped
                    untranslated = _untranslated(by_index[i].get("text", ""), by_index[i].get("text_en", ""))
                    if count_syllables(text, lang) > budget * 1.25 + 2 and not untranslated:
                        logging.info(f"Review ({lang}) L_{i}: fix over budget, kept the draft")
                        continue
                    if text != by_index[i].get("text"):
                        logging.info(f"Review ({lang}) L_{i}: '{by_index[i].get('text')}' -> '{text}'")
                        by_index[i]["text"] = text
                        # The variants predate the fix and may miss what it restored
                        by_index[i]["variants"] = [text]
            except Exception as e:
                logging.warning(f"LLM: Translation review failed for window {w_start}: {e}")

        return [by_index[i] for i in order]

    def refine_translation_by_duration(
        self,
        original_text: str,
        current_text: str,
        actual_dur: float,
        target_dur: float,
        glossary: Dict,
        terms: Optional[Dict] = None,
        context_before: str = "",
        context_after: str = "",
        speaker: str = "",
        media: str = "",
        language: str = "",
    ) -> str:
        """Refines text based on actual audio duration feedback."""
        delta = actual_dur - target_dur
        status = "TOO LONG" if delta > 0 else "TOO SHORT"

        # Don't refine if diff is negligible (< 0.5s or < 10%)
        if abs(delta) < 0.5 and abs(delta / (target_dur + 0.1)) < 0.1:
            return current_text

        logging.info(f"LLM: Refining '{current_text}' ({status}: {actual_dur:.2f}s vs {target_dur:.2f}s)")

        system_prompt = T_REFINE_DURATION.format(
            original_text=original_text,
            current_text=current_text,
            actual_duration=round(actual_dur, 2),
            target_duration=round(target_dur, 2),
            status=status,
            delta=round(delta, 2),
            glossary=json.dumps(glossary, ensure_ascii=False),
            terms=json.dumps(terms, ensure_ascii=False) if terms else "(none)",
            context_before=context_before,
            context_after=context_after,
            speaker=speaker or "unknown",
            media=media or "unknown",
            language=LANG_MAP.get(language, language) or "the target language",
        )

        t0 = time.perf_counter()
        # Refinement is Priority 1 (High)
        # Using temperature 0.3 for more stable/grammatical results
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Adjust this line."}]
        res = self._run_inference(messages, priority=1, max_tokens=200, temperature=0.3)
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
