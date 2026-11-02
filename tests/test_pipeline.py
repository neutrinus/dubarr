import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from core.pipeline import DubbingPipeline


class TestDubbingPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_tts = MagicMock()
        self.mock_diar = MagicMock()
        self.mock_whisper = MagicMock()
        self.pipeline = DubbingPipeline(
            llm_manager=self.mock_llm,
            tts_manager=self.mock_tts,
            diar_manager=self.mock_diar,
            whisper_manager=self.mock_whisper,
            target_langs=["pl"],
            debug_mode=False,
        )

    def test_create_script(self):
        # Two practically continuous lines of the same speaker are merged
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 4.0}]
        trans = [{"start": 0.5, "end": 1.5, "text": "Hello"}, {"start": 1.6, "end": 3.5, "text": "World"}]

        script = self.pipeline._create_script(diar, trans)

        # Merged result: the pause is only 0.1 s
        self.assertEqual(len(script), 1)
        self.assertEqual(script[0]["text_en"], "Hello World")
        self.assertEqual(script[0]["speaker"], "SPEAKER_01")
        self.assertEqual(script[0]["start"], 0.5)
        self.assertEqual(script[0]["end"], 3.5)

    def test_create_script_does_not_merge_across_a_pause(self):
        # A 0.5 s pause stays split: speaking over it drifts out of sync with the original
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 4.0}]
        trans = [{"start": 0.5, "end": 1.5, "text": "Hello"}, {"start": 2.0, "end": 3.5, "text": "World"}]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual([s["text_en"] for s in script], ["Hello", "World"])

    def test_create_script_does_not_merge_beyond_the_duration_cap(self):
        # Back-to-back lines are still left separate when the block would get too long to stay in sync
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 16.0}]
        trans = [{"start": 0.0, "end": 9.0, "text": "First long part"}, {"start": 9.1, "end": 15.0, "text": "Second part"}]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual(len(script), 2)

    def test_create_script_different_speakers(self):
        diar = [{"speaker": "A", "start": 0.0, "end": 2.0}, {"speaker": "B", "start": 2.0, "end": 4.0}]
        trans = [{"start": 0.5, "end": 1.5, "text": "Hello"}, {"start": 2.5, "end": 3.5, "text": "World"}]
        script = self.pipeline._create_script(diar, trans)
        self.assertEqual(len(script), 2)
        self.assertEqual(script[0]["speaker"], "A")
        self.assertEqual(script[1]["speaker"], "B")

    def test_create_script_dominant_speaker(self):
        # Segment touches A briefly but is mostly spoken by B
        diar = [{"speaker": "A", "start": 0.0, "end": 1.2}, {"speaker": "B", "start": 1.2, "end": 5.0}]
        trans = [{"start": 1.0, "end": 4.0, "text": "Mostly B"}]
        script = self.pipeline._create_script(diar, trans)
        self.assertEqual(script[0]["speaker"], "B")

    def test_create_script_no_overlap(self):
        diar = [{"speaker": "A", "start": 10.0, "end": 12.0}]
        trans = [{"start": 0.0, "end": 1.0, "text": "Nobody"}]
        script = self.pipeline._create_script(diar, trans)
        self.assertEqual(script[0]["speaker"], "unknown")

    def test_create_script_slack_after(self):
        diar = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 2.0, "end": 7.55},
            {"speaker": "A", "start": 7.55, "end": 20.0},
        ]
        trans = [
            {"start": 0.0, "end": 1.0, "text": "One"},  # 0.5s gap to B -> 0.35s slack
            {"start": 1.5, "end": 3.0, "text": "Two"},  # 4.0s gap -> capped
            {"start": 7.0, "end": 7.5, "text": "Three"},  # 0.1s gap to A -> no slack
            {"start": 7.6, "end": 9.0, "text": "Four"},  # last segment -> capped
        ]
        with patch("core.pipeline.SLACK_AFTER_MAX", 1.0), patch("core.pipeline.SLACK_GUARD", 0.15):
            script = self.pipeline._create_script(diar, trans)
        self.assertEqual(len(script), 4)
        self.assertAlmostEqual(script[0]["max_end"], 1.35)
        self.assertAlmostEqual(script[1]["max_end"], 4.0)
        self.assertAlmostEqual(script[2]["max_end"], 7.5)
        self.assertAlmostEqual(script[3]["max_end"], 10.0)

    def test_cleanup_debug(self):
        with patch("shutil.rmtree") as mock_rm, patch("os.path.exists", return_value=True), patch("os.makedirs") as mock_mk:
            res = self.pipeline._cleanup_debug("test.mp4")
            mock_rm.assert_called_once()
            # Since debug_mode is False, it shouldn't call makedirs
            mock_mk.assert_not_called()
            self.assertTrue("debug_test" in res)

    def test_extract_refs_returns_golden_refs(self):
        script = [
            {"speaker": "A", "start": 10.0, "end": 18.0, "text_en": "This is a long and clean line of dialogue here"},
            {"speaker": "A", "start": 20.0, "end": 21.0, "text_en": "Short"},
        ]
        self.pipeline.speaker_refs = {"OLD": "/tmp/old.wav"}
        self.pipeline.temp_dir = tempfile.mkdtemp()
        with (
            patch("core.pipeline.run_cmd"),
            patch("core.pipeline.measure_zcr", return_value=0.05),
            patch("core.pipeline.shutil.move"),
        ):
            refs = self.pipeline._extract_refs(script, "vocals.wav", "/tmp/ddir")

        self.assertEqual(list(refs.keys()), ["A"])
        self.assertTrue(refs["A"].endswith("ref_A.wav"))
        self.assertIs(refs, self.pipeline.speaker_refs)
        with open(os.path.join(self.pipeline.temp_dir, "ref_A.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), script[0]["text_en"])
        shutil.rmtree(self.pipeline.temp_dir)


if __name__ == "__main__":
    unittest.main()


class TestSubtitleDiscovery(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.video = os.path.join(self.dir, "Show - S07E02 - The Well WEBDL-1080p.mkv")
        self.pipeline = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def touch(self, suffix):
        path = os.path.splitext(self.video)[0] + suffix
        open(path, "w").close()
        return path

    def test_prefers_full_english_over_hearing_impaired_and_skips_target_language(self):
        self.touch(".pl.srt")
        self.touch(".en.forced.srt")
        hi = self.touch(".en.hi.srt")
        self.assertEqual(self.pipeline._find_external_subtitles(self.video), hi)
        en = self.touch(".en.srt")
        self.assertEqual(self.pipeline._find_external_subtitles(self.video), en)
        plain = self.touch(".srt")
        self.assertEqual(self.pipeline._find_external_subtitles(self.video), plain)

    def test_only_target_language_subtitles(self):
        self.touch(".pl.srt")
        self.touch("-thumb.srt")
        self.assertIsNone(self.pipeline._find_external_subtitles(self.video))
