import unittest
from unittest.mock import MagicMock, patch
from core.tts_manager import TTSManager
import os


class TestTTSManager(unittest.TestCase):
    def setUp(self):
        self.mock_mode_patcher = patch("core.tts_manager.MOCK_MODE", False)
        self.mock_mode_patcher.start()

        self.temp_dir = "/tmp/tts_test"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.manager = TTSManager(
            inference_lock=None,
            temp_dir=self.temp_dir,
            speaker_refs={"SPK1": "golden.wav"},
            abort_event=MagicMock(),
            backend="xtts",
        )
        self.manager.engine = MagicMock()
        self.manager.status = "READY"

    def tearDown(self):
        self.mock_mode_patcher.stop()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_select_reference_prefers_golden_over_dynamic(self):
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment") as mock_extract,
            patch("os.path.exists", side_effect=lambda p: p == "golden.wav"),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK1", "index": 0, "start": 0, "end": 8}, "vocals.wav"
            )
            self.assertEqual(voice_type, "GOLDEN")
            self.assertEqual(ref, "golden.wav")
            mock_extract.assert_not_called()

    def test_select_reference_dynamic_without_golden(self):
        dyn_path = os.path.join(self.temp_dir, "dyn_0.wav")
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment"),
            patch("core.tts_manager.measure_zcr", return_value=0.1),
            patch("os.path.getsize", return_value=10000),
            patch("os.path.exists", side_effect=lambda p: p == dyn_path),
            patch("core.tts_manager.shutil.copy"),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK2", "index": 0, "start": 0, "end": 5}, "vocals.wav"
            )
            self.assertEqual(voice_type, "DYNAMIC")
            self.assertEqual(ref, dyn_path)
            self.assertIn("SPK2", self.manager.last_good_samples)

    def test_select_reference_skips_dynamic_for_short_line(self):
        self.manager.last_good_samples["SPK2"] = "last_good_SPK2.wav"
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment") as mock_extract,
            patch("os.path.exists", side_effect=lambda p: p == "last_good_SPK2.wav"),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK2", "index": 0, "start": 0, "end": 1.5}, "vocals.wav"
            )
            self.assertEqual(voice_type, "LAST_GOOD")
            self.assertEqual(ref, "last_good_SPK2.wav")
            mock_extract.assert_not_called()

    def test_select_reference_falls_back_to_other_golden(self):
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment") as mock_extract,
            patch("os.path.exists", side_effect=lambda p: p == "golden.wav"),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK2", "index": 0, "start": 0, "end": 1}, "vocals.wav"
            )
            self.assertEqual(voice_type, "FALLBACK_GOLDEN")
            self.assertEqual(ref, "golden.wav")
            mock_extract.assert_not_called()

    def test_omnivoice_prefers_clean_line_clip(self):
        self.manager.backend = "omnivoice"
        dyn_path = os.path.join(self.temp_dir, "dyn_0.wav")
        item = {"speaker": "SPK1", "index": 0, "start": 0, "end": 5, "source_text": "Get out of here!"}
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment"),
            patch("core.tts_manager.measure_zcr", return_value=0.1),
            patch("os.path.getsize", return_value=10000),
            patch("core.tts_manager.shutil.copy"),
            patch("os.path.exists", side_effect=lambda p: p in ("golden.wav", dyn_path)),
        ):
            ref, voice_type = self.manager._select_reference(item, "vocals.wav")
        self.assertEqual(voice_type, "DYNAMIC")
        self.assertEqual(ref, dyn_path)
        with open(os.path.join(self.temp_dir, "dyn_0.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "Get out of here!")

    def test_omnivoice_uses_golden_for_short_line(self):
        self.manager.backend = "omnivoice"
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment") as mock_extract,
            patch("os.path.exists", side_effect=lambda p: p == "golden.wav"),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK1", "index": 0, "start": 0, "end": 1.5}, "vocals.wav"
            )
        self.assertEqual(voice_type, "GOLDEN")
        mock_extract.assert_not_called()

    def test_synthesize_item_passes_slot_duration(self):
        item = {"speaker": "SPK1", "index": 0, "text": "Hello", "start": 1.0, "end": 3.5}
        with (
            patch.object(self.manager, "_select_reference", return_value=("ref.wav", "GOLDEN")),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=5000),
        ):
            self.manager._synthesize_item(item, "pl", "vocals.wav", [])
        self.assertEqual(self.manager.engine.synthesize.call_args.kwargs["max_duration"], 2.5)

    def test_set_speaker_refs_resets_last_good(self):
        self.manager.last_good_samples["SPK1"] = "old.wav"
        self.manager.set_speaker_refs({"SPK3": "ref_SPK3.wav"})
        self.assertEqual(self.manager.speaker_refs, {"SPK3": "ref_SPK3.wav"})
        self.assertEqual(self.manager.last_good_samples, {})

    def test_synthesize_item_success(self):
        item = {"speaker": "SPK1", "index": 0, "text": "Hello", "start": 0, "end": 2}

        with (
            patch.object(self.manager, "_select_reference", return_value=("ref.wav", "GOLDEN")),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=5000),
        ):
            res = self.manager._synthesize_item(item, "en", "vocals.wav", [])
            assert res is not None
            assert res["payload"]["voice_type"] == "GOLDEN"
            self.manager.engine.synthesize.assert_called_once()

    def test_synthesize_sync(self):
        item = {"speaker": "SPK1", "index": 0, "text": "Hello", "start": 0, "end": 2}

        with patch.object(self.manager, "_synthesize_item") as mock_synth:
            mock_synth.return_value = {"payload": {"raw_path": "out.wav", "voice_type": "GOLDEN"}, "duration": 1.5}
            res = self.manager.synthesize_sync(item, "en", "vocals.wav", [])
            self.assertEqual(res["audio_path"], "out.wav")
            self.assertEqual(res["voice_type"], "GOLDEN")
