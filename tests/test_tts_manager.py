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
            inference_lock=None, temp_dir=self.temp_dir, speaker_refs={"SPK1": "golden.wav"}, abort_event=MagicMock()
        )
        self.manager.engine = MagicMock()
        self.manager.status = "READY"

    def tearDown(self):
        self.mock_mode_patcher.stop()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_select_reference_golden_fallback(self):
        # Case where dynamic extraction and last good fail
        with (
            patch("core.audio.extract_clean_segment"),
            patch("utils.measure_zcr", return_value=0.5),
            patch("os.path.getsize", return_value=1000),
            patch("os.remove"),
            patch("os.path.exists", side_effect=lambda p: p == "golden.wav" or "dyn" in p),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK1", "index": 0, "start": 0, "end": 1}, "vocals.wav"
            )
            self.assertEqual(voice_type, "GOLDEN")
            self.assertEqual(ref, "golden.wav")

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
