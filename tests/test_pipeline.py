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
        # Scenario: Two segments from same speaker close to each other should be merged
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 2.0}, {"speaker": "SPEAKER_01", "start": 2.5, "end": 4.0}]
        trans = [{"start": 0.5, "end": 1.5, "text": "Hello"}, {"start": 2.0, "end": 3.5, "text": "World"}]

        script = self.pipeline._create_script(diar, trans)

        # Merged result: gap is 2.0 - 1.5 = 0.5 < 1.0
        self.assertEqual(len(script), 1)
        self.assertEqual(script[0]["text_en"], "Hello World")
        self.assertEqual(script[0]["speaker"], "SPEAKER_01")
        self.assertEqual(script[0]["start"], 0.5)
        self.assertEqual(script[0]["end"], 3.5)

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


if __name__ == "__main__":
    unittest.main()
