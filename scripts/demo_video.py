"""Builds a short before/after demo from a dubbed file: one line with the original audio, then the same line dubbed.

Used by scripts/e2e_test.py (the CI E2E job publishes the result for the README), and usable by hand:
    python scripts/demo_video.py dubbed.mkv demo.mp4 --start 12.3 --end 21.0 --langs pl
"""

import argparse
import json
import os
import subprocess
import tempfile

FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf",
]
LANG_NAMES = {"pl": "Polish", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "en": "English"}

# A line shorter than this is too short to judge the voice; a longer one makes the demo drag
MIN_LINE, MAX_LINE = 4.0, 12.0


def pick_window(placed, duration, pad_before=0.4, pad_after=0.6):
    """Picks the [start, end] of one line, covering both the original speech and its dub.

    `placed` are the pipeline's placed segments of one language. Prefers the first line of a comfortable length.
    """
    spans = [(min(p["orig_start"], p["start"]), max(p["orig_end"], p["end"])) for p in placed]
    good = [s for s in spans if MIN_LINE <= s[1] - s[0] <= MAX_LINE]
    if good:
        start, end = good[0]
    elif spans:
        start, end = spans[0][0], min(spans[0][1], spans[0][0] + MAX_LINE)
    else:
        start, end = 0.0, min(duration, MAX_LINE)
    return max(0.0, start - pad_before), min(duration, end + pad_after)


def audio_index(path, title):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "a", "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    streams = json.loads(res.stdout)["streams"]
    for i, s in enumerate(streams):
        if title is None and not s.get("tags", {}).get("title", "").startswith("AI - "):
            return i
        if title is not None and s.get("tags", {}).get("title") == title:
            return i
    return None


def _label_filter(text):
    font = next((f for f in FONTS if os.path.exists(f)), None)
    if not font:
        return ""
    text = text.replace(":", r"\:").replace("'", "")
    return (
        f",drawtext=fontfile='{font}':text='{text}':fontcolor=white:fontsize=h/14:box=1:boxcolor=black@0.55"
        ":boxborderw=12:x=(w-text_w)/2:y=h-th-h/12"
    )


def _part(src, start, end, track, label, out):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
            "-map", "0:v:0", "-map", f"0:a:{track}",
            "-vf", "scale=-2:360,fps=25,format=yuv420p" + _label_filter(label),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000",
            out, "-y",
        ],
        check=True,
        timeout=300,
    )  # fmt: skip


def build_demo(src, out_mp4, start, end, langs, thumbnail=None):
    """Writes an MP4 (H.264 + AAC, plays in browsers and on phones): original line, then each dubbed version."""
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        tracks = [(audio_index(src, None), "ORIGINAL")]
        for lang in langs:
            name = LANG_NAMES.get(lang, lang)
            tracks.append((audio_index(src, f"AI - {name}"), f"AI DUBBING ({name})"))
        for n, (track, label) in enumerate(tracks):
            if track is None:
                continue
            part = os.path.join(tmp, f"part{n}.mp4")
            _part(src, start, end, track, label, part)
            parts.append(part)
        if not parts:
            raise RuntimeError("no audio tracks to build a demo from")
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing,
             "-c", "copy", "-movflags", "+faststart", out_mp4, "-y"],
            check=True,
            timeout=120,
        )  # fmt: skip
    if thumbnail:
        middle = (end - start) * 1.5  # inside the first dubbed part, so the frame shows the dubbing label
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{middle:.3f}", "-i", out_mp4, "-frames:v", "1", "-q:v", "3", thumbnail, "-y"],
            check=True,
            timeout=60,
        )
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--langs", default="pl")
    ap.add_argument("--thumbnail")
    args = ap.parse_args()
    build_demo(args.src, args.out, args.start, args.end, args.langs.split(","), args.thumbnail)


if __name__ == "__main__":
    main()
