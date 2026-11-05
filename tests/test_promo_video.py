import json
import os
import subprocess
import tempfile
import unittest

from scripts.promo_video import audio_track, build_promo, original_track, parse_windows


def make_source(path):
    """A 6 s file with an original track plus Polish, French and German dubs, tagged like the pipeline tags them."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=d=6:s=320x180",
            "-f", "lavfi", "-i", "sine=f=440:d=6",
            "-f", "lavfi", "-i", "sine=f=330:d=6",
            "-f", "lavfi", "-i", "sine=f=220:d=6",
            "-f", "lavfi", "-i", "sine=f=110:d=6",
            "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a", "-map", "4:a",
            "-c:v", "mpeg4", "-c:a", "aac",
            "-metadata:s:a:0", "title=AI - Polish", "-metadata:s:a:0", "language=pol",
            "-metadata:s:a:1", "title=AI - French", "-metadata:s:a:1", "language=fra",
            "-metadata:s:a:2", "title=AI - German", "-metadata:s:a:2", "language=ger",
            "-metadata:s:a:3", "language=und",
            path, "-y",
        ],
        check=True,
    )  # fmt: skip


class TestTracks(unittest.TestCase):
    def test_finds_the_dub_by_title_and_the_original_by_absence(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.mkv")
            make_source(src)
            self.assertEqual(audio_track(src, "pl"), 0)
            self.assertEqual(audio_track(src, "fr"), 1)
            self.assertEqual(audio_track(src, "de"), 2)
            self.assertEqual(audio_track(src, "es"), None)
            self.assertEqual(original_track(src, ["pl", "fr", "de"]), 3)


class TestParseWindows(unittest.TestCase):
    def test_reads_start_end_pairs(self):
        self.assertEqual(parse_windows("0-13.1,13.6-29"), [(0.0, 13.1), (13.6, 29.0)])


class TestBuildPromo(unittest.TestCase):
    def test_every_window_in_every_language(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, thumb = (os.path.join(d, n) for n in ("src.mkv", "promo.mp4", "promo.jpg"))
            make_source(src)
            build_promo(src, out, [(0.0, 2.0), (3.0, 5.0)], ["pl", "fr", "de"], thumb)

            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", out],
                capture_output=True, text=True, check=True,
            )  # fmt: skip
            meta = json.loads(probe.stdout)
            # intro 4 s + 2 windows x (original + 3 dubs) x 2 s + outro 5 s
            self.assertAlmostEqual(float(meta["format"]["duration"]), 25.0, delta=0.5)
            self.assertEqual(sorted(s["codec_name"] for s in meta["streams"]), ["aac", "h264"])
            self.assertTrue(os.path.getsize(thumb) > 0)
