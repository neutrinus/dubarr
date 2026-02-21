import unittest
from unittest.mock import patch
from infrastructure.ffmpeg import FFmpegWrapper


class TestFFmpeg(unittest.TestCase):
    @patch("subprocess.run")
    def test_extract_audio(self, mock_run):
        FFmpegWrapper.extract_audio("input.mp4", "output.wav")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("input.mp4", args)
        self.assertIn("output.wav", args)

    @patch("subprocess.check_output")
    def test_get_duration(self, mock_check_output):
        mock_check_output.return_value = b"123.45\n"
        dur = FFmpegWrapper.get_duration("file.wav")
        self.assertEqual(dur, 123.45)
        mock_check_output.assert_called_once()

    @patch("subprocess.run")
    def test_convert_audio(self, mock_run):
        FFmpegWrapper.convert_audio("in.wav", "out.wav", ac=1, ar=24000)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("24000", args)
        self.assertIn("1", args)

    @patch("subprocess.run")
    def test_get_metadata(self, mock_run):
        mock_run.return_value.stdout = '{"format": {"duration": "10"}}'
        res = FFmpegWrapper.get_metadata("video.mp4")
        self.assertEqual(res["format"]["duration"], "10")
