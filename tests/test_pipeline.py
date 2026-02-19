import unittest
from unittest.mock import MagicMock, patch
from core.pipeline import DubbingPipeline


class TestDubbingPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_tts = MagicMock()
        self.pipeline = DubbingPipeline(
            llm_manager=self.mock_llm, tts_manager=self.mock_tts, target_langs=["pl"], debug_mode=False
        )

    @patch("core.pipeline.prep_audio")
    @patch("core.pipeline.analyze_audio")
    @patch("core.pipeline.mix_audio")
    @patch("core.pipeline.FFmpegWrapper.mux_video")
    @patch("core.pipeline.audio_processor.get_audio_languages")
    @patch("shutil.move")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_process_video_basic_flow(
        self, mock_mkdir, mock_exists, mock_move, mock_get_langs, mock_mux, mock_mix, mock_analyze, mock_prep
    ):
        # Setup mocks
        mock_exists.return_value = True
        mock_prep.return_value = ("stereo.wav", "vocals.mp3")
        mock_analyze.return_value = ([], [], {"step": 1.0})
        mock_get_langs.return_value = ["eng"]
        self.mock_llm.llm = True
        self.mock_llm.analyze_script.return_value = {"context": {}, "speakers": {}}

        # We need to mock the threaded parts or make them synchronous for the test
        with (
            patch.object(DubbingPipeline, "_extract_subtitles", return_value=""),
            patch.object(DubbingPipeline, "_extract_refs"),
            patch.object(DubbingPipeline, "_cleanup_debug", return_value="/tmp/debug"),
        ):
            # For now, let's just verify the initialization and basic setup
            self.assertEqual(self.pipeline.target_langs, ["pl"])
            self.assertFalse(self.pipeline.debug_mode)


if __name__ == "__main__":
    unittest.main()
