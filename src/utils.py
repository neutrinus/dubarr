import json
import os
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


# Release tags that follow the episode title in typical *arr file names
_RELEASE_JUNK = re.compile(
    r"\b(?:480p|576p|720p|1080p|2160p|4k|uhd|hdtv|web[- ]?dl|webrip|bluray|bdrip|dvdrip|remux|"
    r"x264|x265|h ?264|h ?265|hevc|avc|xvid|divx|aac|ac3|dts|ddp?5[.\s]?1|truehd|atmos|"
    r"proper|repack|real|internal|multi|dual)\b.*$",
    re.IGNORECASE,
)


def parse_show_context(path: str) -> str:
    """Best-effort series/episode context from a media file name, e.g. for the translation prompts.

    Handles the usual *arr layouts ("Show (2015) - S01E02 - Title HDTV-720p.mkv") and dotted names
    ("Show.Name.S01E02.1080p.mkv"). Returns an empty string when the name carries no episode info.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(
        r"^(?:(?P<show>.+?)[\s._-]+)?s(?P<season>\d{1,2})e(?P<episode>\d{1,3})(?:[\s._-]+(?P<title>.*))?$",
        stem,
        re.I,
    )
    if not match:
        return ""

    show = re.sub(r"[._]+", " ", match.group("show") or "").strip(" -")
    if not show:
        # "S01E01 Title.mkv" inside ".../Show (2015)/Season 1/": the series name is in the folder
        for parent in reversed(os.path.dirname(os.path.abspath(path)).split(os.sep)):
            if not parent or re.fullmatch(r"(?:season|series|s)\s*\d+|specials?", parent, re.I):
                continue
            show = parent.strip()
            break

    episode = int(match.group("episode"))
    parts = [show] if show else []
    parts.append(f"season {int(match.group('season'))}, episode {episode}")

    title = re.sub(r"[._]+", " ", match.group("title") or "").strip(" -")
    title = _RELEASE_JUNK.sub("", title).strip(" -")
    if title:
        parts.append(f'"{title}"')

    return ", ".join(parts)


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
