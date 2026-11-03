import difflib
import unicodedata
import json
import re
import wave
import struct
import logging
import syllables
import subprocess

try:
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def parse_json(txt: str) -> dict:
    try:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            return json.loads(txt[s : e + 1])
    except Exception as e:
        logging.warning(f"Failed to parse JSON: {e}")
    return {}


def clean_srt(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        if "-->" in line:
            continue
        if line.strip().isdigit():
            continue
        if not line.strip():
            continue
        lines.append(line.strip())
    return " ".join(lines)[:15000]


def measure_zcr(path: str) -> float:
    """Measures Zero Crossing Rate to detect static/screeching."""
    try:
        with wave.open(path, "r") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return 0.0
            frames = wf.readframes(n_frames)
            samples = struct.unpack(f"{n_frames}h", frames)
            zc = 0
            for i in range(1, len(samples)):
                if (samples[i - 1] > 0 and samples[i] <= 0) or (samples[i - 1] <= 0 and samples[i] > 0):
                    zc += 1
            return zc / len(samples)
    except Exception:
        return 0.0


def count_syllables(text: str, lang: str = "pl") -> int:
    if not text:
        return 0
    try:
        count = syllables.estimate(text)
        if lang == "pl":
            vowels = "aeiouyąęó"
            manual_count = sum(1 for char in text.lower() if char in vowels)
            return max(count, manual_count)
        return max(1, count)
    except Exception:
        text = text.lower()
        vowels = "aeiouy"
        if lang == "pl":
            vowels += "ąęó"
        count = sum(1 for char in text if char in vowels)
        return max(1, count)


def clean_output(text: str, target_langs: list) -> str:
    if not text:
        return ""
    text = text.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    text = re.sub(
        r"^(?:translation|thought|analysis|final_text|final|corrected|result|output|text|analysis):\s*",
        "",
        text,
        flags=re.I | re.M,
    )
    text = re.sub(r"\((?:note|translation|explanation|corrected|original).*?\)", "", text, flags=re.I)
    text = re.split(r"\n?(?:note|explanation|comment|uwaga|output|corrected):", text, flags=re.I)[0]

    text = text.split("\n")[0].strip().strip('"').strip("*").strip()

    if HAS_LANGDETECT and text and len(text) > 5:
        try:
            is_latin_target = any(lang in target_langs for lang in ["pl", "en", "de", "es", "fr", "it"])
            if is_latin_target:
                if re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text):
                    logging.warning(f"Language Guard: Detected CJK characters in Latin target '{text}'. Rejecting.")
                    return ""
        except Exception as e:
            logging.warning(f"Langdetect error: {e}")

    return text


def run_cmd(cmd, desc):
    try:
        logging.debug(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error {desc}: {e.stderr}")
        if e.stdout:
            logging.error(f"Stdout {desc}: {e.stdout}")
        raise


def _letters(text: str) -> str:
    """Lower-case letters and digits only: punctuation, hyphens of phonetic spellings and spaces do not count."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def text_similarity(expected: str, heard: str) -> float:
    """Similarity (0..1) of a take's text and what an ASR heard in it, on letters only.

    Tolerant of punctuation, spacing and spelling ("Ajjs-wejv" vs "Ice Wave" costs a few letters), low for noise,
    screams or a different text.
    """
    a, b = _letters(expected), _letters(heard)
    if not a:
        return 1.0
    if not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# English spellings that sound alike fold to one consonant (both sides of the glossary reading test are English spelling)
_SOUND_FOLDS = (
    ("sch", "sh"),
    ("ck", "k"),
    ("ch", "k"),
    ("ph", "f"),
    ("th", "t"),
    ("x", "ks"),
    ("q", "k"),
    ("c", "k"),
    ("b", "p"),
    ("d", "t"),
    ("g", "k"),
    ("z", "s"),
    ("v", "f"),
    ("w", "f"),
)


def consonant_skeleton(text: str) -> str:
    """The consonants of an English spelling, voiced and voiceless pairs folded ("Razorback" -> "rsrpk").

    ``j`` and ``y`` count as vowels: a Polish "j" is an English "y", and the two spellings swap freely.
    """
    t = unicodedata.normalize("NFKD", (text or "").lower())
    t = "".join(ch for ch in t if ch.isascii() and ch.isalpha())
    t = re.sub(r"(.)\1+", r"\1", t)  # doubled letters are one sound ("Sallad")
    for a, b in _SOUND_FOLDS:
        t = t.replace(a, b)
    return re.sub(r"[aeiouyj]", "", t)


def respelling_heard_as(term: str, heard: str) -> bool:
    """Whether what a respelling sounds like ("Rayorbeck") keeps the consonants of the term ("Razorback").

    A respelling that drops or changes a consonant names something else; one change is allowed per 6 consonants
    (Spec 006, 3).
    """
    a, b = consonant_skeleton(term), consonant_skeleton(heard)
    if not a:
        return True
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    missing = sum(i2 - i1 for op, i1, i2, _j1, _j2 in ops if op in ("replace", "delete"))
    return missing <= len(a) // 6
