import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from core.whisper_engine import WhisperManager, _word_bounds


def _seg(start, end, text="x", words=None, avg_logprob=-0.2):
    return SimpleNamespace(start=start, end=end, text=text, words=words, avg_logprob=avg_logprob, no_speech_prob=0.01)


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

        with patch("core.whisper_engine.MOCK_MODE", False):
            res = mgr.transcribe("a.wav")

        kwargs = mgr.model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["word_timestamps"])
        self.assertTrue(kwargs["vad_filter"])
        self.assertEqual(len(res), 1)
        self.assertEqual((res[0]["start"], res[0]["end"]), (0.6, 2.5))
        self.assertEqual(res[0]["text"], "Hello there")


if __name__ == "__main__":
    unittest.main()
