import unittest
from unittest.mock import MagicMock, patch
from core.audio import prep_audio, analyze_audio, get_audio_languages, extract_clean_segment


class TestAudio(unittest.TestCase):
    def test_prep_audio_mock(self):
        with (
            patch("infrastructure.ffmpeg.FFmpegWrapper.extract_audio"),
            patch("shutil.copy"),
            patch("core.audio.MOCK_MODE", True),
        ):
            a, v = prep_audio("video.mp4")
            self.assertTrue(a.endswith("orig.wav"))
            self.assertTrue(v.endswith("vocals.mp3"))

    def test_analyze_audio(self):
        diar_mock = MagicMock()
        whisper_mock = MagicMock()

        diar_mock.diarize.return_value = [{"speaker": "A", "start": 0, "end": 1}]
        whisper_mock.transcribe.return_value = [{"text": "Hello", "start": 0, "end": 1}]

        with patch("infrastructure.ffmpeg.FFmpegWrapper.convert_audio"), patch("core.gpu_manager.GPUManager.force_gc"):
            diar, trans, durs = analyze_audio("vocals.wav", diar_mock, whisper_mock)

            self.assertEqual(len(diar), 1)
            self.assertEqual(trans[0]["text"], "Hello")
            self.assertIn("2a. Diarization", durs)
            self.assertIn("2b. Transcription", durs)

    def test_get_audio_languages(self):
        meta = {
            "streams": [
                {"codec_type": "video"},
                {"codec_type": "audio", "tags": {"language": "eng"}},
                {"codec_type": "audio", "tags": {"language": "pol"}},
            ]
        }
        with patch("infrastructure.ffmpeg.FFmpegWrapper.get_metadata", return_value=meta):
            langs = get_audio_languages("video.mkv")
            self.assertEqual(langs, ["eng", "pol"])

    def test_extract_clean_segment(self):
        with patch("subprocess.run") as mock_run:
            extract_clean_segment("in.wav", 10.0, 15.0, "out.wav")
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            self.assertIn("10.0", args)
            self.assertIn("5.0", args)  # duration
