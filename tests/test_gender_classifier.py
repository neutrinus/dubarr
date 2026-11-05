import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.gender_classifier import GenderClassifier, speaker_clips
from core.pipeline import DubbingPipeline


class TestGenderClassifier(unittest.TestCase):
    def test_mean_of_the_clips(self):
        clf = GenderClassifier(model_name="m")
        probs = [{"female": 0.2, "male": 0.8, "child": 0.0}, {"female": 0.0, "male": 1.0, "child": 0.0}]
        with patch.object(clf, "probabilities", side_effect=probs):
            verdict, confidence = clf.classify([1, 2])
        self.assertEqual(verdict, "male")
        self.assertAlmostEqual(confidence, 0.9)

    def test_a_child_or_nothing_tells_no_gender(self):
        clf = GenderClassifier(model_name="m")
        with patch.object(clf, "probabilities", return_value={"female": 0.1, "male": 0.1, "child": 0.8}):
            self.assertEqual(clf.classify([1])[0], None)
        with patch.object(clf, "probabilities", return_value=None):
            self.assertEqual(clf.classify([1]), (None, 0.0))

    def test_off_and_mock_mode(self):
        self.assertFalse(GenderClassifier(model_name="off").enabled)
        self.assertFalse(GenderClassifier(model_name="m").enabled)  # tests run in MOCK_MODE
        self.assertIsNone(GenderClassifier(model_name="m").probabilities(np.zeros(32000)))

    def test_calm_lines_longest_first(self):
        samples = np.arange(16000 * 40, dtype=np.float32)
        script = [
            {"speaker": "A", "start": 0.0, "end": 2.0, "energy": "calm"},
            {"speaker": "A", "start": 5.0, "end": 25.0, "energy": "calm"},  # cut to 10 s
            {"speaker": "A", "start": 30.0, "end": 36.0, "energy": "shout"},
            {"speaker": "A", "start": 37.0, "end": 38.0, "energy": "calm"},  # too short
            {"speaker": "B", "start": 2.0, "end": 5.0},
        ]
        clips = speaker_clips(script, samples, "A")
        self.assertEqual([len(c) / 16000 for c in clips], [10.0, 2.0])
        # Only shouting lines: those are better than nothing
        self.assertEqual(len(speaker_clips(script[2:3], samples, "A")), 1)
        self.assertEqual(speaker_clips(script, None, "A"), [])


class TestLastResort(unittest.TestCase):
    def check(self, speaker_info, verdict, script=None, media=None):
        pipe = DubbingPipeline.__new__(DubbingPipeline)
        pipe.target_langs = ["pl"]
        pipe.temp_dir = "/tmp"
        pipe.media_info = media or {}
        pipe.speaker_info = speaker_info
        clf = MagicMock(enabled=True)
        clf.classify.return_value = verdict
        with (
            patch("core.pipeline.MOCK_MODE", False),
            patch("core.pipeline.GenderClassifier", return_value=clf),
            patch("core.pipeline.script_check.load_samples", return_value=np.zeros(16000)),
            patch("core.pipeline.speaker_pitch", return_value=None),
        ):
            pipe._check_genders({spk: ["r.wav"] for spk in speaker_info}, script or [], "vocals.mp3")
        clf.shutdown.assert_called_once()
        return pipe.speaker_info

    def test_decides_for_an_unnamed_speaker(self):
        # MythBusters' narrator: "Unknown, female" from the analysis
        info = self.check({"S6": {"name": "Unknown", "gender": "female"}}, ("male", 0.97))
        self.assertEqual(info["S6"]["gender"], "male")

    def test_not_sure_enough(self):
        info = self.check({"S6": {"name": "", "gender": "female"}}, ("male", 0.7))
        self.assertEqual(info["S6"]["gender"], "female")

    def test_a_named_speaker_keeps_the_analysis(self):
        info = self.check({"S5": {"name": "Alton Brown", "gender": "male"}}, ("female", 0.92))
        self.assertEqual(info["S5"]["gender"], "male")

    def test_a_doubted_name_lets_the_voice_decide(self):
        script = [{"speaker": "S3", "start": 0.0, "end": 3.0, "speaker_doubt": "Chris Rose introduces Chris Rose"}]
        info = self.check({"S3": {"name": "Chris Rose", "gender": "male"}}, ("female", 0.92), script)
        self.assertEqual(info["S3"]["gender"], "female")
        self.assertEqual(info["S3"]["name"], "")  # "Chris Rose (female)" would mislead the translator

    def test_the_cast_wins_over_a_less_sure_voice(self):
        media = {"cast": [{"character": "Kirk", "actor": "William Shatner", "gender": "male"}]}
        info = self.check({"S7": {"name": "Unknown Kirk", "gender": "male"}}, ("female", 0.93), media=media)
        self.assertEqual(info["S7"]["gender"], "male")

    def test_a_sure_voice_wins_over_a_swapped_name(self):
        # MythBusters S2012E19: the analysis named Grant "Kari Byron"; the classifier heard a man (0.96)
        media = {"cast": [{"character": "Kari Byron", "actor": "Kari Byron", "gender": "female"}]}
        info = self.check({"S7": {"name": "Kari Byron", "gender": "female"}}, ("male", 0.96), media=media)
        self.assertEqual(info["S7"]["gender"], "male")
        self.assertEqual(info["S7"]["name"], "")
        self.assertTrue(info["S7"]["gender_measured"])

    def test_the_subtitles_grammar_wins_over_a_sure_voice(self):
        script = [
            {"speaker": "S2", "start": float(i), "end": i + 1.0, "subtitles": {"pl": f"Zrobiłem to {i}."}} for i in range(4)
        ]
        info = self.check({"S2": {"name": "", "gender": ""}}, ("female", 0.99), script)
        self.assertEqual(info["S2"]["gender"], "male")


if __name__ == "__main__":
    unittest.main()
