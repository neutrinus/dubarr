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

    def test_cleanup_debug(self):
        with patch("shutil.rmtree") as mock_rm, patch("os.path.exists", return_value=True), patch("os.makedirs") as mock_mk:
            res = self.pipeline._cleanup_debug("test.mp4")
            mock_rm.assert_called_once()
            # Since debug_mode is False, it shouldn't call makedirs
            mock_mk.assert_not_called()
            self.assertTrue("debug_test" in res)


if __name__ == "__main__":
    unittest.main()
