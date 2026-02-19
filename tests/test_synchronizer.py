import unittest
from unittest.mock import MagicMock, patch
from core.synchronizer import SegmentSynchronizer


class TestSynchronizer(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_tts = MagicMock()
        self.temp_dir = "/tmp"
        self.sync = SegmentSynchronizer(self.mock_llm, self.mock_tts, self.temp_dir)

    def test_perfect_match(self):
        # Scenario: Duration target = 5.0, TTS produces 5.1 (acceptable)
        segment = {"index": 1, "text_en": "Hello", "start": 0, "end": 5, "speaker": "A"}

        self.mock_tts.synthesize_sync.return_value = {"audio_path": "fake.wav", "duration": 5.1, "voice_type": "MOCK"}

        # We need to mock FFmpegWrapper.get_duration inside synchronizer because it checks physical file
        with patch("core.synchronizer.FFmpegWrapper.get_duration", return_value=5.1):
            with patch("shutil.copy"):  # Mock copy so we don't need real files
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {})

        self.assertEqual(res["status"], "ACCEPTED")
        self.mock_llm.refine_translation_by_duration.assert_not_called()

    def test_too_long_refinement(self):
        # Scenario: Target=5.0. Attempt 1=7.0 (Too long). LLM shortens. Attempt 2=5.2 (OK).
        segment = {"index": 1, "text_en": "Hello long", "start": 0, "end": 5, "speaker": "A"}

        # TTS returns first long, then ok
        self.mock_tts.synthesize_sync.side_effect = [
            {"audio_path": "long.wav", "duration": 7.0, "voice_type": "MOCK"},
            {"audio_path": "short.wav", "duration": 5.2, "voice_type": "MOCK"},
        ]

        self.mock_llm.refine_translation_by_duration.return_value = "Short text"

        with patch("core.synchronizer.FFmpegWrapper.get_duration", side_effect=[7.0, 5.2]):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {})

        self.assertEqual(res["status"], "ACCEPTED")
        self.assertEqual(res["final_text"], "Short text")
        self.mock_llm.refine_translation_by_duration.assert_called_once()

    def test_fallback(self):
        # Scenario: Target=5.0. All attempts fail (too long). Should return best (shortest delta).
        segment = {"index": 1, "text_en": "Hello fail", "start": 0, "end": 5, "speaker": "A"}

        # Attempt 1: 8.0s (Delta +3)
        # Attempt 2: 7.0s (Delta +2) -> LLM refined
        # Attempt 3: 7.5s (Delta +2.5) -> LLM refined badly

        self.mock_tts.synthesize_sync.side_effect = [
            {"audio_path": "a1.wav", "duration": 8.0, "voice_type": "MOCK"},
            {"audio_path": "a2.wav", "duration": 7.0, "voice_type": "MOCK"},
            {"audio_path": "a3.wav", "duration": 7.5, "voice_type": "MOCK"},
        ]

        self.mock_llm.refine_translation_by_duration.side_effect = ["Text 2", "Text 3"]

        with patch("core.synchronizer.FFmpegWrapper.get_duration", side_effect=[8.0, 7.0, 7.5]):
            with patch("shutil.copy"):
                res = self.sync.process_segment(segment, "pl", "voc.wav", [], {}, attempt_limit=3)

        self.assertEqual(res["status"], "FALLBACK")
        self.assertEqual(res["duration"], 7.0)  # Should pick Attempt 2 (smallest delta)
        self.assertEqual(res["final_text"], "Text 2")
