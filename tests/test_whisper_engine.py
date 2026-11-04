import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from core.whisper_engine import WhisperManager, _gaps, _plausible, _split_at_pauses, _word_bounds


def _seg(start, end, text="x", words=None, avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2):
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=words,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


def _word(start, end, word="w", probability=0.9):
    return SimpleNamespace(start=start, end=end, word=word, probability=probability)


class TestWordBounds(unittest.TestCase):
    def test_tightens_to_words(self):
        self.assertEqual(_word_bounds(_seg(0.0, 5.0, words=[_word(0.8, 1.2), _word(1.3, 3.9)])), (0.8, 3.9))

    def test_falls_back_without_words(self):
        self.assertEqual(_word_bounds(_seg(0.0, 5.0, words=None)), (0.0, 5.0))
        self.assertEqual(_word_bounds(_seg(0.0, 5.0, words=[])), (0.0, 5.0))

    def test_falls_back_on_degenerate_words(self):
        self.assertEqual(_word_bounds(_seg(0.0, 5.0, words=[_word(2.0, 2.0)])), (0.0, 5.0))


class TestRunTranscribe(unittest.TestCase):
    def test_requests_word_timestamps_and_uses_them(self):
        mgr = WhisperManager(inference_lock=None)
        mgr.model = MagicMock()
        segs = [
            _seg(0.0, 4.0, " Hello there ", words=[_word(0.6, 1.0), _word(1.1, 2.5)]),
            _seg(4.0, 6.0, "garbage", avg_logprob=-2.0),
        ]
        mgr.model.transcribe.return_value = (iter(segs), SimpleNamespace(duration=6.0, language="en"))

        with patch("core.whisper_engine.MOCK_MODE", False), patch("core.whisper_engine.ASR_GAP_PASS", False):
            res = mgr.transcribe("a.wav")

        kwargs = mgr.model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["word_timestamps"])
        self.assertTrue(kwargs["vad_filter"])
        self.assertEqual(len(res), 1)
        self.assertEqual((res[0]["start"], res[0]["end"]), (0.6, 2.5))
        self.assertEqual(res[0]["text"], "Hello there")


class TestGapPass(unittest.TestCase):
    def test_gaps_between_and_after_segments(self):
        segs = [{"start": 1.0, "end": 4.0}, {"start": 4.5, "end": 10.0}, {"start": 20.0, "end": 25.0}]
        self.assertEqual(_gaps(segs, 30.0, 2.0), [(10.0, 20.0), (25.0, 30.0)])

    def test_only_confident_speech_is_plausible(self):
        self.assertTrue(_plausible(_seg(0, 2, " Let the bot battle begin!")))
        self.assertFalse(_plausible(_seg(0, 2, " Thank you.")))
        self.assertFalse(_plausible(_seg(0, 2, " Instruction.")))
        self.assertFalse(_plausible(_seg(0, 2, " Three, two, one.", no_speech_prob=0.7)))
        self.assertFalse(_plausible(_seg(0, 2, " Go!", avg_logprob=-1.2)))
        self.assertFalse(_plausible(_seg(0, 2, " Go! Go! Go! Go! Go! Go!", compression_ratio=3.0)))
        unsure = [SimpleNamespace(start=0, end=1, word="w", probability=0.2)] * 3
        self.assertFalse(_plausible(_seg(0, 2, " Three two one", words=unsure)))

    def test_gap_is_transcribed_without_vad_and_offset(self):
        mgr = WhisperManager(inference_lock=None)
        mgr.model = MagicMock()
        first = [
            _seg(
                0.0,
                10.0,
                " Ready for battle.",
                words=[_word(0.5 + 1.5 * i, 1.9 + 1.5 * i) for i in range(5)] + [_word(8.0, 9.0)],
            )
        ]
        gap = [_seg(1.0, 3.0, " Let the bot battle begin!", words=[_word(1.2, 2.8)]), _seg(5.0, 8.0, " Thank you.")]
        mgr.model.transcribe.side_effect = [
            (iter(first), SimpleNamespace(duration=20.0, language="en")),
            (iter(gap), SimpleNamespace(duration=11.0, language="en")),
        ]

        with patch("core.whisper_engine.MOCK_MODE", False), patch("core.whisper_engine.ASR_GAP_PASS", True):
            with patch("core.whisper_engine.decode_audio", return_value=[0.0] * (16000 * 20)):
                res = mgr.transcribe("a.wav")

        kwargs = mgr.model.transcribe.call_args_list[1].kwargs
        self.assertFalse(kwargs["vad_filter"])
        self.assertEqual(kwargs["language"], "en")
        self.assertEqual([r["text"] for r in res], ["Ready for battle.", "Let the bot battle begin!"])
        self.assertAlmostEqual(res[1]["start"], 9.0 + 1.2)


class TestSplitAtPauses(unittest.TestCase):
    def test_a_segment_without_long_pauses_stays_whole(self):
        seg = _seg(0.0, 3.0, " Ready for battle.", words=[_word(0.2, 0.6), _word(0.7, 1.0), _word(1.1, 1.6)])
        self.assertEqual(_split_at_pauses(seg, 2.0), [(0.2, 1.6, None, seg.words)])

    def test_long_pauses_split_the_words(self):
        words = [
            _word(2198.0, 2262.0, " Warrior"),
            _word(2262.1, 2262.5, " Clan"),
            _word(2262.6, 2263.0, " is"),
            _word(2339.0, 2339.4, " that"),
            _word(2339.5, 2342.0, " ends"),
        ]
        parts = _split_at_pauses(_seg(2198.0, 2342.4, " Warrior Clan is that ends", words=words), 2.0)
        self.assertEqual([p[2] for p in parts], ["Warrior Clan is", "that ends"])
        # the first word was stretched over the silence before it: only its last 1.5 s count
        self.assertEqual((parts[0][0], parts[0][1]), (2260.5, 2263.0))
        self.assertEqual((parts[1][0], parts[1][1]), (2339.0, 2342.0))

    def test_a_stretched_word_alone_is_trimmed(self):
        parts = _split_at_pauses(_seg(540.0, 565.0, " Ready for battle.", words=[_word(542.4, 564.7)]), 2.0)
        self.assertEqual([(p[0], p[1], p[2]) for p in parts], [(563.2, 564.7, None)])

    def test_the_gap_pass_hears_the_pause_inside_a_segment(self):
        mgr = WhisperManager(inference_lock=None)
        mgr.model = MagicMock()
        first = [
            _seg(
                0.0,
                30.0,
                " Warrior Clan knocked out",
                words=[_word(0.0, 1.0, " Warrior Clan"), _word(28.0, 29.0, " knocked out")],
            )
        ]
        gap = [_seg(2.0, 6.0, " Nightmare is in trouble!", words=[_word(2.0, 3.0), _word(3.1, 6.0)])]
        mgr.model.transcribe.side_effect = [
            (iter(first), SimpleNamespace(duration=30.0, language="en")),
            (iter(gap), SimpleNamespace(duration=27.0, language="en")),
        ]

        with patch("core.whisper_engine.MOCK_MODE", False), patch("core.whisper_engine.ASR_GAP_PASS", True):
            with patch("core.whisper_engine.decode_audio", return_value=[0.0] * (16000 * 30)):
                res = mgr.transcribe("a.wav")

        self.assertEqual([r["text"] for r in res], ["Warrior Clan", "Nightmare is in trouble!", "knocked out"])
        self.assertAlmostEqual(res[1]["start"], 1.0 + 2.0)

    def test_without_the_gap_pass_nothing_is_split(self):
        mgr = WhisperManager(inference_lock=None)
        mgr.model = MagicMock()
        first = [_seg(0.0, 30.0, " Warrior Clan knocked out", words=[_word(0.0, 1.0), _word(28.0, 29.0)])]
        mgr.model.transcribe.return_value = (iter(first), SimpleNamespace(duration=30.0, language="en"))
        with patch("core.whisper_engine.MOCK_MODE", False), patch("core.whisper_engine.ASR_GAP_PASS", False):
            res = mgr.transcribe("a.wav")
        self.assertEqual([(r["start"], r["end"], r["text"]) for r in res], [(0.0, 29.0, "Warrior Clan knocked out")])


class TestTakeCheckDevice(unittest.TestCase):
    def test_take_check_model_stays_off_the_gpu_by_default(self):
        mgr = WhisperManager(inference_lock=None)
        with patch("core.whisper_engine.WhisperModel") as model, patch("core.whisper_engine.GPUManager.get_best_gpu") as gpu:
            with patch("core.whisper_engine.TAKE_CHECK_DEVICE", "cpu"), patch("core.whisper_engine.TAKE_CHECK_THREADS", 12):
                mgr._load_take_model()
        gpu.assert_not_called()
        self.assertEqual(model.call_args.kwargs["device"], "cpu")
        # Every core, not faster-whisper's default of 4 threads
        self.assertEqual(model.call_args.kwargs["cpu_threads"], 12)

    def test_auto_needs_room_for_the_tts_as_well(self):
        mgr = WhisperManager(inference_lock=None)
        with (
            patch("core.whisper_engine.WhisperModel"),
            patch("core.whisper_engine.GPUManager.get_best_gpu", return_value="cpu") as gpu,
        ):
            with patch("core.whisper_engine.TAKE_CHECK_DEVICE", "auto"):
                mgr._load_take_model()
        self.assertGreaterEqual(gpu.call_args.kwargs["needed_mb"], 8000)


if __name__ == "__main__":
    unittest.main()
