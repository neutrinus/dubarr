import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch
from core.synchronizer import SegmentSynchronizer


class TestSynchronizer(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_tts = MagicMock()
        self.temp_dir = "/tmp"
        self.sync = SegmentSynchronizer(self.mock_llm, self.mock_tts, self.temp_dir)

    def test_perfect_match(self):
        # Scenario: Duration target = 5.0, TTS produces 5.0 (perfect)
        segment = {"index": 0, "text_en": "Hello", "start": 0, "end": 5, "speaker": "A"}

        self.mock_tts.synthesize_sync.return_value = {"audio_path": "fake.wav", "duration": 5.0, "voice_type": "MOCK"}

        # We need to mock FFmpegWrapper.get_duration inside synchronizer because it checks physical file
        with patch("core.synchronizer.FFmpegWrapper.get_duration", return_value=5.0):
            with patch("shutil.copy"):  # Mock copy so we don't need real files
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {})

        self.assertEqual(res["status"], "ACCEPTED")
        self.mock_llm.refine_translation_by_duration.assert_not_called()

    def test_accepts_overrun_into_trailing_silence(self):
        # Target=5.0 but 0.8s of silence follows; TTS produces 5.6 -> fits without refinement
        segment = {"index": 0, "text_en": "Hello", "start": 0, "end": 5, "max_end": 5.8, "speaker": "A"}

        self.mock_tts.synthesize_sync.return_value = {"audio_path": "fake.wav", "duration": 5.6, "voice_type": "MOCK"}

        with patch("core.synchronizer.FFmpegWrapper.get_duration", return_value=5.6):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {})

        self.assertEqual(res["status"], "ACCEPTED")
        self.mock_llm.refine_translation_by_duration.assert_not_called()

    def test_small_overrun_is_accepted_and_speed_up_later(self):
        # Target 5.0, TTS produces 5.2: within MAX_ACCEPT_SPEEDUP, the mix speeds the clip up instead of the
        # LLM deleting words
        segment = {"index": 0, "text_en": "Hello", "start": 0, "end": 5, "speaker": "A"}

        self.mock_tts.synthesize_sync.return_value = {"audio_path": "fake.wav", "duration": 5.2, "voice_type": "MOCK"}

        with patch("core.synchronizer.FFmpegWrapper.get_duration", return_value=5.2):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {}, attempt_limit=1)

        self.assertEqual(res["status"], "ACCEPTED")
        self.mock_llm.refine_translation_by_duration.assert_not_called()

    def test_large_overrun_is_still_refined(self):
        segment = {"index": 0, "text_en": "Hello", "start": 0, "end": 5, "speaker": "A"}
        self.mock_tts.synthesize_sync.return_value = {"audio_path": "fake.wav", "duration": 5.9, "voice_type": "MOCK"}

        with patch("core.synchronizer.FFmpegWrapper.get_duration", return_value=5.9):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {}, attempt_limit=1)

        self.assertEqual(res["status"], "FORCED_SHORT_FALLBACK")

    def test_too_long_refinement(self):
        # Scenario: Target=5.0. Attempt 1=7.0 (Too long). LLM shortens. Attempt 2=5.0 (OK).
        segment = {"index": 0, "text_en": "Hello long", "start": 0, "end": 5, "speaker": "A"}

        # TTS returns first long, then ok
        self.mock_tts.synthesize_sync.side_effect = [
            {"audio_path": "long.wav", "duration": 7.0, "voice_type": "MOCK"},
            {"audio_path": "short.wav", "duration": 5.0, "voice_type": "MOCK"},
        ]

        self.mock_llm.refine_translation_by_duration.return_value = "Short text"

        with patch("core.synchronizer.FFmpegWrapper.get_duration", side_effect=[7.0, 5.0]):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {})

        self.assertEqual(res["status"], "ACCEPTED")
        self.assertEqual(res["final_text"], "Short text")
        self.mock_llm.refine_translation_by_duration.assert_called_once()

    def test_fallback(self):
        # Scenario: Target=5.0. All attempts fail (too long). Should return best (shortest duration).
        segment = {"index": 0, "text_en": "Hello fail", "start": 0, "end": 5, "speaker": "A"}

        # Attempt 1: 8.0s
        # Attempt 2: 7.0s -> LLM refined
        # Attempt 3: 7.5s -> LLM refined badly

        self.mock_tts.synthesize_sync.side_effect = [
            {"audio_path": "a1.wav", "duration": 8.0, "voice_type": "MOCK"},
            {"audio_path": "a2.wav", "duration": 7.0, "voice_type": "MOCK"},
            {"audio_path": "a3.wav", "duration": 7.5, "voice_type": "MOCK"},
        ]

        self.mock_llm.refine_translation_by_duration.side_effect = ["Text 2", "Text 3"]

        with patch("core.synchronizer.FFmpegWrapper.get_duration", side_effect=[8.0, 7.0, 7.5]):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {}, attempt_limit=3)

        self.assertEqual(res["status"], "FORCED_SHORT_FALLBACK")
        self.assertEqual(res["duration"], 7.0)  # Should pick Attempt 2 (shortest)
        self.assertEqual(res["final_text"], "Text 2")

    def test_failed_synthesis_returns_silent_segment_not_source(self):
        segment = {"index": 7, "text_en": "Hello", "start": 2, "end": 4.5, "speaker": "A"}
        self.mock_tts.synthesize_sync.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            sync = SegmentSynchronizer(self.mock_llm, self.mock_tts, temp_dir)
            res = sync.process_segment(segment, "pl", "voc.wav", [], {})

            self.assertEqual(res["status"], "FAILED_FALLBACK")
            self.assertNotEqual(res["audio_path"], "voc.wav")
            with wave.open(res["audio_path"], "rb") as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.getnframes(), 60000)
