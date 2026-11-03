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


class TestSyncContentKeeping(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.tts = MagicMock()
        self.sync = SegmentSynchronizer(self.llm, self.tts, "/tmp")

    def run_segment(self, segment, durations, **kwargs):
        with patch("core.synchronizer.FFmpegWrapper.get_duration", side_effect=durations):
            with patch("shutil.copy"), patch("os.path.exists", return_value=False):
                return self.sync.process_segment(segment, "pl", "voc.wav", [], {}, **kwargs)

    def test_swallowed_take_is_synthesized_again_with_another_reference(self):
        # "A niebieski kwadrat." came back as 0.64 s of audio: mostly silence trimmed by OmniVoice
        segment = {
            "index": 28,
            "text_en": "And the blue square!",
            "text": "A niebieski kwadrat.",
            "start": 0,
            "end": 1.6,
            "speaker": "A",
        }
        self.tts.synthesize_sync.side_effect = [
            {"audio_path": "a.wav", "duration": 0.6, "voice_type": "DYNAMIC", "ref": "dyn.wav"},
            {"audio_path": "b.wav", "duration": 1.4, "voice_type": "GOLDEN", "ref": "gold.wav"},
        ]

        res = self.run_segment(segment, [0.64, 1.45])

        self.assertEqual(res["status"], "ACCEPTED")
        self.assertEqual(res["duration"], 1.45)
        retry_item = self.tts.synthesize_sync.call_args_list[1].args[0]
        self.assertEqual(retry_item["skip_refs"], ["DYNAMIC"])
        self.llm.refine_translation_by_duration.assert_not_called()

    def test_take_swallowed_on_every_reference_is_not_reworded(self):
        segment = {
            "index": 3,
            "text_en": "x",
            "text": "Mark DeVitz, geniusz elektroniki.",
            "start": 0,
            "end": 4.0,
            "speaker": "A",
        }
        self.tts.synthesize_sync.return_value = {"audio_path": "a.wav", "duration": 0.5, "voice_type": "GOLDEN", "ref": "g"}

        res = self.run_segment(segment, [0.5, 0.5, 0.5])

        self.assertEqual(self.tts.synthesize_sync.call_count, 3)
        self.llm.refine_translation_by_duration.assert_not_called()
        self.assertNotEqual(res["status"], "ACCEPTED")

    def test_too_short_take_is_expanded(self):
        segment = {
            "index": 0,
            "text_en": "Part of the reason is two mini-bots",
            "text": "To Plan Eks.",
            "start": 0,
            "end": 5.0,
            "speaker": "A",
        }
        self.tts.synthesize_sync.side_effect = [
            {"audio_path": "a.wav", "duration": 1.5, "voice_type": "GOLDEN", "ref": "g"},
            {"audio_path": "b.wav", "duration": 4.6, "voice_type": "GOLDEN", "ref": "g"},
        ]
        self.llm.refine_translation_by_duration.return_value = "To Plan Eks, częściowo dlatego, że ma dwa miniboty."

        res = self.run_segment(segment, [1.5, 4.6])

        self.assertEqual(res["status"], "ACCEPTED")
        self.assertEqual(res["final_text"], "To Plan Eks, częściowo dlatego, że ma dwa miniboty.")
        self.assertEqual(self.llm.refine_translation_by_duration.call_args.kwargs["target_dur"], 5.0)

    def test_short_original_stays_short_when_nothing_can_be_added(self):
        segment = {"index": 0, "text_en": "Wow!", "text": "Łał!", "start": 0, "end": 2.0, "speaker": "A"}
        self.tts.synthesize_sync.return_value = {"audio_path": "a.wav", "duration": 0.6, "voice_type": "GOLDEN", "ref": "g"}
        self.llm.refine_translation_by_duration.return_value = "Łał!"

        res = self.run_segment(segment, [0.6])

        self.assertEqual(res["status"], "ACCEPTED")
        self.assertEqual(self.tts.synthesize_sync.call_count, 1)

    def test_refinement_sees_neighbouring_translations(self):
        script = [
            {"text_en": "Before", "translations": {"pl": "Przed"}},
            {"text_en": "Now"},
            {"text_en": "After", "translations": {"pl": "Po"}},
        ]
        self.assertEqual(SegmentSynchronizer._neighbours(script, 1, "pl"), ("Before => Przed", "After => Po"))


class TestLengthVariants(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.tts = MagicMock()
        self.sync = SegmentSynchronizer(self.llm, self.tts, "/tmp")
        self.segment = {
            "index": 0,
            "speaker": "A",
            "text_en": "x",
            "text": "Normalna wersja kwestii.",
            "variants": ["Normalna wersja kwestii.", "Krótka.", "Dłuższa i pełniejsza wersja tej kwestii."],
            "start": 0,
            "end": 3.0,
            "max_end": 3.5,
        }

    def run_segment(self, durations):
        with patch("core.synchronizer.FFmpegWrapper.get_duration", side_effect=durations):
            with patch("shutil.copy"), patch("os.path.exists", return_value=False):
                return self.sync.process_segment(self.segment, "pl", "voc.wav", [], {})

    def test_best_fitting_variant_is_synthesized_first(self):
        self.tts.estimate_durations.return_value = [2.0, 1.0, 3.1]
        self.tts.synthesize_sync.return_value = {"audio_path": "a.wav", "duration": 3.0, "voice_type": "GOLDEN", "ref": "g"}

        res = self.run_segment([3.0])

        self.assertEqual(res["final_text"], "Dłuższa i pełniejsza wersja tej kwestii.")
        self.llm.refine_translation_by_duration.assert_not_called()

    def test_overrun_counts_more_than_falling_short(self):
        self.assertEqual(SegmentSynchronizer._miss(2.6, 3.0, 3.5), SegmentSynchronizer._miss(3.4, 3.0, 3.5))
        self.assertGreater(SegmentSynchronizer._miss(4.0, 3.0, 3.5), SegmentSynchronizer._miss(2.0, 3.0, 3.5))

    def test_too_long_take_tries_the_shorter_variant_before_the_llm(self):
        self.tts.estimate_durations.return_value = [2.9, 1.5, 4.5]
        self.tts.synthesize_sync.return_value = {"audio_path": "a.wav", "duration": 4.0, "voice_type": "GOLDEN", "ref": "g"}

        res = self.run_segment([4.5, 2.8])

        self.assertEqual(res["status"], "ACCEPTED")
        self.assertEqual(res["final_text"], "Krótka.")
        self.llm.refine_translation_by_duration.assert_not_called()

    def test_without_an_estimate_the_main_translation_is_used(self):
        self.tts.estimate_durations.return_value = None
        self.tts.synthesize_sync.return_value = {"audio_path": "a.wav", "duration": 3.0, "voice_type": "GOLDEN", "ref": "g"}

        res = self.run_segment([3.0])

        self.assertEqual(res["final_text"], "Normalna wersja kwestii.")
