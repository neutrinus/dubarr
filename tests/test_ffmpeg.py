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
    def test_mux_keeps_original_streams(self, mock_run):
        FFmpegWrapper.mux_video("in.mkv", [("pl.wav", "pl", "Polish"), ("de.wav", "de", "German")], "out.mkv", "aac", "192k")
        args = mock_run.call_args[0][0]
        joined = " ".join(args)
        # dubbed tracks first, then the original audio, subtitles and attachments
        self.assertIn("-map 1:a -map 2:a -map 0:a -map 0:s? -map 0:t?", joined)
        # only the dubbed tracks are encoded
        self.assertIn("-c copy -c:a:0 aac -b:a:0 192k -c:a:1 aac -b:a:1 192k", joined)
        self.assertIn("language=pol", args)
        self.assertIn("language=ger", args)
        self.assertIn("title=AI - Polish", args)
        self.assertFalse(any(a.startswith("handler_name=") for a in args))

    @patch("subprocess.run")
    def test_mux_mp4_names_the_track_handler(self, mock_run):
        FFmpegWrapper.mux_video("in.mp4", [("pl.wav", "pl", "Polish")], "out.mp4")
        self.assertIn("handler_name=AI - Polish", mock_run.call_args[0][0])

    @patch("subprocess.run")
    def test_mean_volume(self, mock_run):
        mock_run.return_value.stderr = "[Parsed_volumedetect_0] mean_volume: -23.4 dB\n[...] max_volume: -3.0 dB"
        self.assertEqual(FFmpegWrapper.mean_volume("a.mp3", 1.0, 2.0), -23.4)
        mock_run.return_value.stderr = "[Parsed_volumedetect_0] mean_volume: -inf dB"
        self.assertIsNone(FFmpegWrapper.mean_volume("a.mp3", 1.0, 2.0))

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


class TestMuxCodecs(unittest.TestCase):
    @patch("subprocess.run")
    def test_encodes_only_the_new_tracks(self, mock_run):
        FFmpegWrapper.mux_video("in.mkv", [("pl.wav", "pl", "Polish"), ("de.wav", "de", "German")], "out.mkv", "ac3", "384k")
        args = mock_run.call_args[0][0]
        joined = " ".join(args)
        self.assertIn("-c copy", joined)  # original audio, subtitles and attachments untouched
        self.assertIn("-c:a:0 ac3 -b:a:0 384k", joined)
        self.assertIn("-c:a:1 ac3 -b:a:1 384k", joined)
        self.assertNotIn("-c:a:2", joined)

    def test_unknown_codec_falls_back_to_aac(self):
        from infrastructure.ffmpeg import audio_codec_args

        self.assertEqual(audio_codec_args("mp3", "192k", ":0"), ["-c:a:0", "aac", "-b:a:0", "192k"])
