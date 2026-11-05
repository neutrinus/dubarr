"""Builds a promo video from a dubbed file: title card, then N fragments of the original, each followed by its dubs.

The source is a file the pipeline dubbed into several languages (one audio track per language, the original kept).
Every fragment is cut with the same [start, end] from every track, so the viewer hears the same seconds in each
language, one after another, with a "PART n/N" badge and a language label burned in.

    python scripts/promo_video.py dubbed.mp4 promo.mp4 --windows 0-13.1,13.6-29.0,29.5-45.5 --langs pl,fr,de

The workflow around it (choosing a public-domain source, checking that each fragment has a different speaker) is in
`docs/promo_video.md`.
"""

import argparse
import json
import os
import subprocess
import tempfile

FONTS_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
LANG_NAMES = {"pl": "Polish", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "en": "English"}
ISO_639_2 = {"pl": "pol", "fr": "fra", "de": "ger", "es": "spa", "it": "ita", "en": "eng"}

WIDTH, HEIGHT, FPS = 1280, 720, 30
BG = "0x0d1117"
ACCENT = "0x4cc2ff"


def font(paths):
    return next((p for p in paths if os.path.exists(p)), None)


def escape(text):
    """Escapes a drawtext value (its parser treats `:` and `%` specially)."""
    return text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")


def audio_streams(path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_streams", "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(res.stdout)["streams"]


def audio_track(path, lang):
    """Index of the dubbed track of `lang`, by its title (MKV) or its language tag (MP4 drops the titles)."""
    streams = audio_streams(path)
    title = f"AI - {LANG_NAMES.get(lang, lang)}"
    for i, stream in enumerate(streams):
        if stream.get("tags", {}).get("title") == title:
            return i
    codes = {ISO_639_2.get(lang, lang), lang}
    for i, stream in enumerate(streams):
        if stream.get("tags", {}).get("language") in codes:
            return i
    return None


def original_track(path, langs):
    """The original audio: the last track that is not one of the dubbed languages (the mux keeps it last)."""
    streams = audio_streams(path)
    dubbed = {ISO_639_2.get(lang, lang) for lang in langs}
    for i in range(len(streams) - 1, -1, -1):
        if streams[i].get("tags", {}).get("language") not in dubbed:
            return i
    return len(streams) - 1 if streams else None


def encode(inputs, video_filter, out, audio=None):
    """One part of the promo, always with the same codec settings so the parts can be joined."""
    args = ["ffmpeg", "-v", "error"] + inputs
    if audio is None:
        # silence for the cards: the second input has to come before any output option
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-vf", video_filter, "-shortest"]
    else:
        # -map disables the automatic stream selection, so the video has to be mapped explicitly
        args += ["-vf", video_filter, "-map", "0:v:0", "-map", f"0:a:{audio}", "-ac", "2", "-ar", "48000"]
    args += [
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", out, "-y",
    ]  # fmt: skip
    subprocess.run(args, check=True, timeout=900)


def card(out, title, subtitle, footer, seconds):
    """A title or closing card: three centred lines of text on a dark background."""
    bold, regular = font(FONTS_BOLD), font(FONTS)
    video_filter = (
        "format=yuv420p,"
        f"drawtext=fontfile='{bold}':text='{escape(title)}':fontcolor=white:fontsize=96:"
        "x=(w-text_w)/2:y=(h-text_h)/2-70,"
        f"drawtext=fontfile='{regular}':text='{escape(subtitle)}':fontcolor={ACCENT}:fontsize=40:"
        "x=(w-text_w)/2:y=(h-text_h)/2+50,"
        f"drawtext=fontfile='{regular}':text='{escape(footer)}':fontcolor=0x9aa4b2:fontsize=28:"
        "x=(w-text_w)/2:y=h-110"
    )
    encode(["-f", "lavfi", "-i", f"color=c={BG}:s={WIDTH}x{HEIGHT}:d={seconds}:r={FPS}"], video_filter, out)


def part(src, start, end, track, label, badge, out):
    """One fragment in one language, with the language label and the "PART n/N" badge."""
    bold, regular = font(FONTS_BOLD), font(FONTS)
    video_filter = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p,"
        "drawbox=x=0:y=ih-150:w=iw:h=150:color=black@0.62:t=fill,"
        f"drawtext=fontfile='{bold}':text='{escape(label)}':fontcolor=white:fontsize=46:x=56:y=h-112,"
        "drawbox=x=44:y=36:w=210:h=64:color=black@0.55:t=fill,"
        f"drawtext=fontfile='{bold}':text='{escape(badge)}':fontcolor={ACCENT}:fontsize=34:"
        "x=44+(210-text_w)/2:y=36+(64-text_h)/2,"
        f"drawtext=fontfile='{regular}':text='dubarr':fontcolor=white@0.75:fontsize=26:x=w-tw-56:y=40"
    )
    encode(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", src], video_filter, out, audio=track)


def build_promo(src, out, windows, langs, thumbnail=None):
    """Writes the promo: title card, then every window in the original and in each language, then the closing card."""
    parts = []
    total = len(windows)
    with tempfile.TemporaryDirectory() as tmp:
        intro = os.path.join(tmp, "00_intro.mp4")
        card(intro, "dubarr", "AI dubbing for your media library", "English  →  Polish  ·  French  ·  German", 4.0)
        parts.append(intro)

        tracks = [(original_track(src, langs), "ORIGINAL", None)]
        for lang in langs:
            tracks.append((audio_track(src, lang), f"AI DUBBING — {LANG_NAMES.get(lang, lang)}", lang))

        n = 0
        for i, (start, end) in enumerate(windows, 1):
            for track, label, _lang in tracks:
                if track is None:
                    continue
                n += 1
                path = os.path.join(tmp, f"{n:02d}_part.mp4")
                part(src, start, end, track, label, f"PART {i}/{total}", path)
                parts.append(path)

        outro = os.path.join(tmp, "99_outro.mp4")
        card(
            outro,
            "runs on your GPU",
            "self-hosted  ·  no cloud APIs  ·  the original audio is kept",
            "github.com/neutrinus/dubarr",
            5.0,
        )
        parts.append(outro)

        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        # Re-encode the join: the parts come from different sources, and a stream copy leaves mismatched extradata
        # behind, which players then decode as garbage.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing,
             "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out, "-y"],
            check=True,
            timeout=1800,
        )  # fmt: skip
    if thumbnail:
        # a frame of the last fragment's original, the widest shot of the source material
        at = sum(end - start for start, end in windows) * (len(windows) - 1) + 4.0
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", out, "-frames:v", "1", "-q:v", "2", thumbnail, "-y"],
            check=True,
            timeout=120,
        )  # fmt: skip
    return out


def parse_windows(text):
    """'0-13.1,13.6-29' -> [(0.0, 13.1), (13.6, 29.0)]."""
    windows = []
    for chunk in text.split(","):
        start, end = chunk.split("-")
        windows.append((float(start), float(end)))
    return windows


def main():
    ap = argparse.ArgumentParser(description="Promo video: the same fragments, the original then every dub.")
    ap.add_argument("src", help="a file dubbed into several languages")
    ap.add_argument("out", help="the promo MP4 to write")
    ap.add_argument("--windows", required=True, help="start-end,start-end,... in seconds")
    ap.add_argument("--langs", default="pl,fr,de")
    ap.add_argument("--thumbnail", help="also write a JPEG frame here")
    args = ap.parse_args()
    build_promo(args.src, args.out, parse_windows(args.windows), args.langs.split(","), args.thumbnail)


if __name__ == "__main__":
    main()
