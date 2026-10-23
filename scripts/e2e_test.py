"""End-to-end test: dub a short real clip with the real (small, CPU) models and check the result.

Runs the full DubbingPipeline in-process (no API, no Docker) and then verifies the output container:
one new audio track per target language, tagged correctly, roughly as long as the video and not silent.
It also reports, without failing on it, which language Whisper hears on the dubbed track.

Model choice and device come from environment variables (see AGENTS.md); in CI the job uses CPU-sized
models. Without HF_TOKEN, diarization is replaced by a single speaker so the rest can still be tested.

Usage:
    python scripts/e2e_test.py --url <video url or path> [--start 60] [--duration 60] [--langs pl]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "https://svs.gsfc.nasa.gov/vis/a010000/a013800/a013868/ISS_HIGH_DEF_FEED.mp4"


def run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)


def fetch_clip(url, start, duration, out_path):
    """Cuts [start, start+duration] out of the source (local path or URL) into a small MKV."""
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            url,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            "scale=-2:360",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-metadata:s:a:0",
            "language=eng",
            out_path,
            "-y",
        ]
    )


def probe(path):
    res = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], timeout=30)
    return json.loads(res.stdout)


def mean_volume(path, audio_index):
    res = subprocess.run(
        ["ffmpeg", "-i", path, "-map", f"0:a:{audio_index}", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", res.stderr)
    return float(m.group(1)) if m else None


def detect_language(path, audio_index, whisper_model):
    """Best-effort: which language does Whisper hear on the dubbed track? Never fails the test."""
    try:
        from faster_whisper import WhisperModel

        wav = path + f".a{audio_index}.wav"
        run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:a:{audio_index}", "-ac", "1", "-ar", "16000", wav, "-y"])
        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav)
        text = " ".join(s.text.strip() for s in segments)
        return info.language, info.language_probability, text
    except Exception as e:  # noqa: BLE001 - informative only
        return None, 0.0, f"language check failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("E2E_VIDEO_URL") or DEFAULT_URL)
    ap.add_argument("--start", type=float, default=float(os.environ.get("E2E_START", 60)))
    ap.add_argument("--duration", type=float, default=float(os.environ.get("E2E_DURATION", 60)))
    ap.add_argument("--langs", default=os.environ.get("TARGET_LANGS", "pl"))
    ap.add_argument("--workdir", default=os.path.join(ROOT, "e2e_out"))
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    clip = os.path.join(args.workdir, "clip.mkv")
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    # Configure the app before anything from src/ is imported (config.py reads env at import time).
    os.environ["MOCK_MODE"] = "0"
    os.environ["TARGET_LANGS"] = ",".join(langs)
    os.environ.setdefault("DATA_DIR", os.path.join(args.workdir, "data"))
    no_diarization = not os.environ.get("HF_TOKEN")
    if no_diarization:
        os.environ["HF_TOKEN"] = "missing"
    sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

    print(f"Fetching {args.duration:.0f}s from {args.url} (offset {args.start:.0f}s)...", flush=True)
    fetch_clip(args.url, args.start, args.duration, clip)
    src_meta = probe(clip)
    src_duration = float(src_meta["format"]["duration"])
    src_audio_count = sum(1 for s in src_meta["streams"] if s["codec_type"] == "audio")

    from config import LANG_MAP, WHISPER_MODEL
    from main import AIDubber

    dubber = AIDubber()
    if no_diarization:
        print("WARNING: HF_TOKEN not set, diarization replaced by a single speaker.", flush=True)
        dubber.diar_manager.diarize = lambda _path: [{"start": 0.0, "end": src_duration, "speaker": "SPEAKER_00"}]
        dubber.diar_manager.shutdown = lambda: None

    t0 = time.perf_counter()
    dubber.process_video(clip)
    elapsed = time.perf_counter() - t0

    # --- Verification ---
    errors, report = [], []
    out_meta = probe(clip)
    audio = [s for s in out_meta["streams"] if s["codec_type"] == "audio"]
    report.append(f"Pipeline time: {elapsed / 60:.1f} min for a {src_duration:.0f}s clip ({', '.join(langs)})")
    report.append(f"Diarization: {'single-speaker stub (no HF_TOKEN)' if no_diarization else 'pyannote'}")

    if len(audio) != src_audio_count + len(langs):
        errors.append(f"expected {src_audio_count + len(langs)} audio tracks, found {len(audio)}")

    for lang in langs:
        title = f"AI - {LANG_MAP.get(lang, lang)}"
        idx = next((i for i, s in enumerate(audio) if s.get("tags", {}).get("title") == title), None)
        if idx is None:
            errors.append(f"no audio track titled '{title}'")
            continue
        tags = audio[idx].get("tags", {})
        if tags.get("language") != lang:
            errors.append(f"track '{title}' has language={tags.get('language')!r}, expected {lang!r}")

        vol = mean_volume(clip, idx)
        if vol is None or vol < -50:
            errors.append(f"track '{title}' looks silent (mean volume {vol} dB)")

        heard, prob, text = detect_language(clip, idx, WHISPER_MODEL)
        report.append(f"[{lang}] mean volume {vol} dB; Whisper hears '{heard}' (p={prob:.2f})")
        report.append(f"[{lang}] transcript: {text[:500]}")

    out_duration = float(out_meta["format"]["duration"])
    if abs(out_duration - src_duration) > 1.5:
        errors.append(f"output duration {out_duration:.1f}s differs from source {src_duration:.1f}s")

    log_path = os.path.join(os.environ["DATA_DIR"], "logs", "processing.log")
    if os.path.exists(log_path):
        log = open(log_path, encoding="utf-8", errors="replace").read()
        for label, pattern in [
            ("segments accepted", r"\] ACCEPTED"),
            ("segments in fallback", r"_FALLBACK: Selection"),
            ("segments failed", r"failed completely|TIMED OUT|All synthesis attempts failed"),
        ]:
            count = len(re.findall(pattern, log))
            report.append(f"{label}: {count}")
            if label == "segments failed" and count:
                errors.append(f"{count} segment(s) failed, see processing.log")

    summary = ["## dubarr E2E (CPU)", "", *[f"- {line}" for line in report], ""]
    summary += ["**Result: PASS**"] if not errors else ["**Result: FAIL**", *[f"- {e}" for e in errors]]
    text = "\n".join(summary)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
