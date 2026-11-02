import unittest
from unittest.mock import MagicMock, patch
from core.audio import prep_audio, analyze_audio, get_audio_languages, extract_clean_segment, clarity_db


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
        diar_mock.embeddings = {"A": [0.1, 0.2]}
        whisper_mock.transcribe.return_value = [{"text": "Hello", "start": 0, "end": 1}]

        with patch("infrastructure.ffmpeg.FFmpegWrapper.convert_audio"), patch("core.gpu_manager.GPUManager.force_gc"):
            diar, trans, durs, prints = analyze_audio("vocals.wav", diar_mock, whisper_mock)
            self.assertEqual(prints, {"A": [0.1, 0.2]})

            self.assertEqual(len(diar), 1)
            self.assertEqual(trans[0]["text"], "Hello")
            self.assertIn("2a. Diarization", durs)
            self.assertIn("2b. Transcription", durs)

    def test_whisper_hears_the_original_mix(self):
        diar_mock, whisper_mock = MagicMock(), MagicMock()
        whisper_mock.transcribe.return_value = []

        with (
            patch("infrastructure.ffmpeg.FFmpegWrapper.convert_audio") as convert,
            patch("core.gpu_manager.GPUManager.force_gc"),
        ):
            with patch("core.audio.ASR_SOURCE", "original"):
                analyze_audio("vocals.mp3", diar_mock, whisper_mock, "orig.wav")

        self.assertEqual(convert.call_args_list[1].args[0], "orig.wav")
        self.assertTrue(whisper_mock.transcribe.call_args.args[0].endswith("asr_mono.wav"))
        self.assertTrue(diar_mock.diarize.call_args.args[0].endswith("mono.wav"))
        self.assertFalse(diar_mock.diarize.call_args.args[0].endswith("asr_mono.wav"))

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

    def test_clarity_is_voice_over_the_rest_of_the_mix(self):
        levels = {"/t/vocals.mp3": -20.0, "/t/no_vocals.mp3": -26.5}
        with patch("core.audio.os.path.exists", return_value=True):
            with patch("core.audio.FFmpegWrapper.mean_volume", side_effect=lambda p, s, d: levels[p]):
                self.assertAlmostEqual(clarity_db("/t/vocals.mp3", 1.0, 3.0), 6.5)
        with patch("core.audio.os.path.exists", side_effect=lambda p: p == "/t/vocals.mp3"):
            self.assertIsNone(clarity_db("/t/vocals.mp3", 1.0, 3.0))

    def test_extract_clean_segment(self):
        with patch("subprocess.run") as mock_run:
            extract_clean_segment("in.wav", 10.0, 15.0, "out.wav")
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            self.assertIn("10.0", args)
            self.assertIn("5.0", args)  # duration


class TestMixAudio(unittest.TestCase):
    def test_speech_past_the_end_does_not_lengthen_the_track(self):
        import os
        import subprocess
        import tempfile

        from config import TEMP_DIR
        from core.audio import mix_audio
        from infrastructure.ffmpeg import FFmpegWrapper

        os.makedirs(TEMP_DIR, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            bg, speech, out = (os.path.join(d, n) for n in ("bg.wav", "speech.wav", "out.wav"))
            for path, seconds in ((bg, 4), (speech, 3)):
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=f=300:d={seconds}", "-ac", "2", path, "-y"],
                    check=True,
                )
            mix_audio(bg, [(speech, 3.0, 3.0)], out)  # speech ends at 6 s, background at 4 s
            self.assertAlmostEqual(FFmpegWrapper.get_duration(out), 4.0, delta=0.2)

    def test_final_limiter_never_reaches_full_scale(self):
        import re
        import subprocess

        from core.audio import FINAL_LIMITER

        stats = subprocess.run(
            [
                "ffmpeg",
                "-nostats",
                "-f",
                "lavfi",
                "-i",
                "sine=f=440:d=3,volume=12",
                "-af",
                f"{FINAL_LIMITER},astats=measure_perchannel=none",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        ).stderr
        peak = float(re.search(r"Peak level dB: (-?[\d.]+)", stats).group(1))
        # alimiter's default auto level pushed the limited mix back to 0 dBFS (S01E03: +0.7 dBTP after AC3)
        self.assertLessEqual(peak, -1.0)


def test_calm_bonus():
    from core.audio import calm_bonus

    assert calm_bonus(-20.0, -20.0) == 5
    assert calm_bonus(-17.0, -20.0) == 0
    assert calm_bonus(-15.0, -20.0) == -10
    assert calm_bonus(-5.0, -20.0) == -40
    assert calm_bonus(None, -20.0) == 0
