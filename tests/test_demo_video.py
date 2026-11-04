import json
import os
import subprocess
import tempfile
import unittest

from scripts.demo_video import build_demo, pick_window


def seg(start, end):
    return {"index": 0, "orig_start": start, "orig_end": end, "start": start, "end": end}


class TestPickWindow(unittest.TestCase):
    def test_prefers_first_line_of_comfortable_length(self):
        placed = [seg(0, 18.4), seg(18.5, 37.7), seg(38.0, 47.6), seg(48.0, 53.2)]
        self.assertEqual(pick_window(placed, 60), (37.6, 48.2))

    def test_covers_the_dub_when_it_runs_longer(self):
        placed = [{"index": 0, "orig_start": 2.0, "orig_end": 7.0, "start": 2.0, "end": 8.0}]
        self.assertEqual(pick_window(placed, 60, 0, 0), (2.0, 8.0))

    def test_long_lines_are_capped_and_no_lines_fall_back(self):
        self.assertEqual(pick_window([seg(0, 30)], 60, 0, 0), (0.0, 12.0))
        self.assertEqual(pick_window([], 5, 0, 0), (0.0, 5.0))


class TestBuildDemo(unittest.TestCase):
    def test_original_then_dub(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, thumb = (os.path.join(d, n) for n in ("src.mkv", "demo.mp4", "demo.jpg"))
            subprocess.run(
                [
                    "ffmpeg", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc=d=6:s=320x180",
                    "-f", "lavfi", "-i", "sine=f=440:d=6",
                    "-f", "lavfi", "-i", "sine=f=220:d=6",
                    "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "mpeg4", "-c:a", "aac",
                    "-metadata:s:a:0", "title=AI - Polish", "-metadata:s:a:0", "language=pl",
                    "-metadata:s:a:1", "language=eng", src, "-y",
                ],
                check=True,
            )  # fmt: skip
            build_demo(src, out, 1.0, 3.0, ["pl"], thumb)

            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", out],
                capture_output=True, text=True, check=True,
            )  # fmt: skip
            meta = json.loads(probe.stdout)
            self.assertAlmostEqual(float(meta["format"]["duration"]), 4.0, delta=0.3)
            self.assertEqual(sorted(s["codec_name"] for s in meta["streams"]), ["aac", "h264"])
            self.assertTrue(os.path.getsize(thumb) > 0)
