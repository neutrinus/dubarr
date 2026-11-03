import os
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

from core.take_judge import TakeJudge, find_artifacts, read_wav_mono

RATE = 24000


def speech(seconds, rate=RATE):
    """A voice-like signal: 150 Hz with harmonics, syllable-like envelope, a little noise."""
    t = np.arange(int(seconds * rate)) / rate
    tone = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    rng = np.random.default_rng(0)
    return (0.08 * tone * envelope + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def write_wav(path, samples, rate=RATE):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class TestArtifacts(unittest.TestCase):
    def test_clean_speech_passes(self):
        self.assertIsNone(find_artifacts(speech(3.0), RATE))

    def test_squeak_is_found(self):
        # 6:53 of the measured run: a high-pitched squeak of about a second inside the take
        s = speech(3.0)
        t = np.arange(int(0.8 * RATE)) / RATE
        s[RATE : RATE + len(t)] += 0.3 * np.sin(2 * np.pi * 6000 * t).astype(np.float32)
        self.assertIn("SQUEAK", find_artifacts(s, RATE))

    def test_short_sibilant_is_not_a_squeak(self):
        s = speech(3.0)
        rng = np.random.default_rng(1)
        hiss = rng.standard_normal(int(0.12 * RATE)).astype(np.float32)
        hiss = np.diff(hiss, prepend=0) * 0.2  # a short "sz": high-pitched noise for 120 ms
        s[RATE : RATE + len(hiss)] += hiss
        self.assertIsNone(find_artifacts(s, RATE))

    def test_clipping_is_found(self):
        s = speech(2.0) * 20
        self.assertIn("CLIPPING", find_artifacts(np.clip(s, -1, 1), RATE))

    def test_long_gap_inside_is_found_but_not_at_the_ends(self):
        quiet = np.zeros(int(1.5 * RATE), dtype=np.float32)
        self.assertIn("GAP", find_artifacts(np.concatenate([speech(1.0), quiet, speech(1.0)]), RATE))
        self.assertIsNone(find_artifacts(np.concatenate([quiet, speech(2.0), quiet]), RATE))

    def test_read_wav(self):
        path = os.path.join(tempfile.mkdtemp(), "t.wav")
        write_wav(path, speech(0.5))
        samples, rate = read_wav_mono(path)
        self.assertEqual((rate, len(samples)), (RATE, RATE // 2))
        self.assertIsNone(read_wav_mono(path + ".missing"))


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "take.wav")
        write_wav(self.path, speech(3.0))
        self.judge = TakeJudge()
        self.text = "Ice Wave próbuje zachować dystans."

    def test_sound_take(self):
        with (
            patch.object(self.judge, "voice_match", return_value=0.82),
            patch.object(self.judge, "text_match", return_value=0.9),
        ):
            res = self.judge.judge(self.text, self.path, "pl", "ref.wav")
        self.assertEqual(res, {"score": 0.9, "problem": None, "voice": 0.82, "text_match": 0.9})

    def test_wrong_voice_halves_the_score(self):
        with (
            patch.object(self.judge, "voice_match", return_value=0.2),
            patch.object(self.judge, "text_match", return_value=0.9),
        ):
            res = self.judge.judge(self.text, self.path, "pl", "ref.wav")
        self.assertIn("WRONG VOICE", res["problem"])
        self.assertEqual(res["score"], 0.45)

    def test_unintelligible(self):
        with (
            patch.object(self.judge, "voice_match", return_value=0.8),
            patch.object(self.judge, "text_match", return_value=0.1),
        ):
            res = self.judge.judge(self.text, self.path, "pl", "ref.wav")
        self.assertIn("UNINTELLIGIBLE", res["problem"])

    def test_without_models_only_the_signal_counts(self):
        # MOCK_MODE (tests): no voiceprint and no CTC model, the take is sound
        res = self.judge.judge(self.text, self.path, "pl", "ref.wav")
        self.assertEqual(res, {"score": 1.0, "problem": None, "voice": None, "text_match": None})

    def test_interjections_skip_the_text_match(self):
        with patch.object(self.judge, "text_match") as text_match:
            self.judge.judge("Wow!", self.path, "pl")
        text_match.assert_not_called()

    def test_signal_only(self):
        judge = TakeJudge(signal_only=True)
        with patch.object(judge, "voice_match") as voice:
            self.assertIsNone(judge.judge(self.text, self.path, "pl", "ref.wav")["problem"])
        voice.assert_not_called()


class TestModels(unittest.TestCase):
    def test_ctc_model_falls_back_to_mms(self):
        judge = TakeJudge()
        loaded = ("model", "processor")
        with (
            patch("core.take_judge.MOCK_MODE", False),
            patch.object(TakeJudge, "_load_ctc", side_effect=[ImportError("pyctcdecode"), loaded]) as load,
        ):
            self.assertEqual(judge._align_model("pl"), loaded)
        self.assertEqual(
            [c.args[0] for c in load.call_args_list], ["jonatasgrosman/wav2vec2-large-xlsr-53-polish", "facebook/mms-1b-all"]
        )

    def test_ctc_processor_is_loaded_without_a_language_model(self):
        transformers = __import__("pytest").importorskip("transformers")
        with (
            patch.object(transformers, "Wav2Vec2Processor") as proc,
            patch.object(transformers, "Wav2Vec2ForCTC") as model,
        ):
            TakeJudge._load_ctc("jonatasgrosman/wav2vec2-large-xlsr-53-polish", "pl")
        proc.from_pretrained.assert_called_once()
        model.from_pretrained.assert_called_once()

    def test_voiceprint_is_computed_from_samples_in_memory(self):
        torch = __import__("pytest").importorskip("torch")
        path = os.path.join(tempfile.mkdtemp(), "take.wav")
        write_wav(path, speech(1.0))
        judge = TakeJudge()
        seen = {}

        def model(inp):
            seen.update(inp)
            return np.array([[3.0, 4.0]])

        with patch.object(judge, "_voice_model", return_value=model):
            emb = judge.voiceprint(path)
        self.assertEqual(seen["sample_rate"], 16000)
        self.assertIsInstance(seen["waveform"], torch.Tensor)
        np.testing.assert_allclose(emb, [0.6, 0.8])


class TestCtcScore(unittest.TestCase):
    def test_fitting_text_scores_high_and_other_text_low(self):
        torch = __import__("pytest").importorskip("torch")
        from core.take_judge import ctc_score

        # 3 symbols + blank (0); frames that clearly say 1, 2, 3
        frames = torch.full((6, 4), -10.0)
        for i, sym in enumerate([1, 0, 2, 0, 3, 0]):
            frames[i, sym] = 0.0
        log_probs = frames.log_softmax(-1)
        self.assertGreater(ctc_score(log_probs, [1, 2, 3], 0), 0.9)
        self.assertLess(ctc_score(log_probs, [3, 2, 1], 0), 0.1)
        self.assertEqual(ctc_score(log_probs[:2], [1, 2, 3], 0), 0.0)


if __name__ == "__main__":
    unittest.main()
