import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from core.whisper_engine import WhisperManager, _gaps, _plausible, _word_bounds


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


def _word(start, end):
    return SimpleNamespace(start=start, end=end, word="w", probability=0.9)


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
        first = [_seg(0.0, 10.0, " Ready for battle.", words=[_word(0.5, 9.0)])]
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


if __name__ == "__main__":
    unittest.main()
