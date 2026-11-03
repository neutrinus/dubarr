import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.tts_manager import TTSManager


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
        )
        self.manager.engine = MagicMock()
        self.manager.status = "READY"

    def tearDown(self):
        self.mock_mode_patcher.stop()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_select_reference_uses_golden_for_long_line(self):
        # Lines longer than LINE_REF_RANGE are not used as their own reference
        with (
            patch("core.tts_manager.audio_processor.extract_clean_segment") as mock_extract,
            patch("os.path.exists", side_effect=lambda p: p == "golden.wav"),
        ):
            ref, voice_type = self.manager._select_reference(
                {"speaker": "SPK1", "index": 0, "start": 0, "end": 12}, "vocals.wav"
            )
            self.assertEqual(voice_type, "GOLDEN")
            self.assertEqual(ref, "golden.wav")
            mock_extract.assert_not_called()

    def test_select_reference_dynamic_without_golden(self):
        dyn_path = os.path.join(self.temp_dir, "dyn__0.wav")
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

    def test_prefers_clean_line_clip(self):
        dyn_path = os.path.join(self.temp_dir, "dyn__0.wav")
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
        with open(os.path.join(self.temp_dir, "dyn__0.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "Get out of here!")

    def test_uses_golden_for_short_line(self):
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
        call = self.manager.engine.synthesize.call_args
        self.assertEqual(call.kwargs["max_duration"], 2.5)
        self.assertTrue(call.args[2].endswith("raw_pl_0.wav"))

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


class TestTTSDeviceSelection(unittest.TestCase):
    def _manager(self):
        return TTSManager(
            inference_lock=None,
            temp_dir="/tmp/tts_test",
            speaker_refs={},
            abort_event=MagicMock(),
        )

    def test_cpu_device_keeps_the_gpu_free(self):
        manager = self._manager()
        with (
            patch("core.tts_manager.MOCK_MODE", False),
            patch("core.tts_manager.TTS_DEVICE", "cpu"),
            patch("core.tts_manager.GPUManager") as gpu_manager,
            patch("core.tts_manager.TTSWrapper") as wrapper,
        ):
            manager.load_engine()

        gpu_manager.get_best_gpu.assert_not_called()
        self.assertEqual(wrapper.call_args.kwargs["gpu_id"], -1)
        self.assertEqual(manager.device, "cpu")

    def test_auto_device_uses_the_gpu_manager(self):
        manager = self._manager()
        with (
            patch("core.tts_manager.MOCK_MODE", False),
            patch("core.tts_manager.TTS_DEVICE", "auto"),
            patch("core.tts_manager.GPUManager") as gpu_manager,
            patch("core.tts_manager.TTSWrapper") as wrapper,
        ):
            gpu_manager.get_best_gpu.return_value = "cuda:1"
            manager.load_engine()

        gpu_manager.get_best_gpu.assert_called_once()
        self.assertEqual(wrapper.call_args.kwargs["gpu_id"], 1)
        self.assertEqual(manager.device, "cuda:1")


class TestChunkedSynthesis(unittest.TestCase):
    def test_short_line_is_one_chunk(self):
        from core.tts_manager import split_for_tts

        self.assertEqual(split_for_tts("Krótka kwestia.", 3.0), [("Krótka kwestia.", 3.0)])

    def test_long_line_is_split_at_sentences(self):
        from core.tts_manager import split_for_tts

        text = "Pierwsze zdanie jest naprawdę długie i opowiada o tym, jak zawodnicy przygotowywali swoje maszyny przez wiele miesięcy. Drugie zdanie również jest długie i wyjaśnia, dlaczego ta walka jest tak ważna dla wszystkich obecnych na arenie. Trzecie zdanie zamyka wypowiedź i zapowiada kolejne starcie."
        chunks = split_for_tts(text, 26.0, max_chunk=10.0)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(c[0] for c in chunks), text)
        self.assertAlmostEqual(sum(c[1] for c in chunks), 26.0, places=1)

    def test_single_long_sentence_is_cut_on_commas_or_words(self):
        from core.tts_manager import split_for_tts

        text = "To jedno bardzo długie zdanie bez kropek, ale z przecinkami, które trzeba pociąć na kawałki."
        chunks = split_for_tts(text, 20.0, max_chunk=5.0)

        self.assertGreater(len(chunks), 1)
        for chunk_text, _ in chunks:
            self.assertLessEqual(len(chunk_text), 80)

    def test_long_sentence_breaks_on_commas(self):
        from core.tts_manager import split_for_tts

        text = (
            "Pierwsza część zdania mówi o jednym, druga część zdania mówi o drugim, "
            "trzecia część zdania mówi o trzecim, czwarta część zdania mówi o czwartym."
        )
        chunks = split_for_tts(text, 20.0, max_chunk=4.0)

        self.assertGreater(len(chunks), 1)
        for chunk_text, _ in chunks[:-1]:
            self.assertTrue(chunk_text.endswith((",", ";", ":", ".")), chunk_text)

    def test_chunks_are_synthesized_separately_and_joined(self):
        manager = TTSManager(
            inference_lock=None,
            temp_dir=tempfile.mkdtemp(),
            speaker_refs={},
            abort_event=MagicMock(),
        )
        self.addCleanup(shutil.rmtree, manager.temp_dir, ignore_errors=True)
        calls = []

        def fake_synthesis(text, ref, out_path, language=None, max_duration=None, target_duration=None):
            calls.append((text, max_duration))
            with open(out_path, "wb") as f:
                f.write(b"RIFF")

        with (
            patch.object(manager, "_run_synthesis", side_effect=fake_synthesis),
            patch.object(manager, "_concat_wavs") as concat,
        ):
            manager._synthesize_chunked(
                "Pierwsze zdanie jest naprawdę długie i opowiada o tym, jak zawodnicy przygotowywali swoje maszyny przez wiele miesięcy. Drugie zdanie również jest długie i wyjaśnia, dlaczego ta walka jest tak ważna dla wszystkich obecnych na arenie. Trzecie zdanie zamyka wypowiedź i zapowiada kolejne starcie.",
                "ref.wav",
                os.path.join(manager.temp_dir, "out.wav"),
                "pl",
                26.0,
                7,
            )

        self.assertEqual(len(calls), len(concat.call_args.args[0]))
        self.assertGreater(len(calls), 1)
        for _, max_duration in calls:
            self.assertLess(max_duration, 26.0)
