"""Fixes after the BattleBots S01E01 run with Whisper large-v3 (repeated words, second-hearing hallucinations,
doubted names)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.pipeline import DubbingPipeline
from core.whisper_engine import WhisperManager

MEDIA = {"cast": [{"actor": "Chris Rose", "character": "Play-By-Play Announcer", "gender": "male"}]}


def _line(start, end, text, speaker="S5", asr="sure"):
    return {"start": start, "end": end, "text_en": text, "speaker": speaker, "asr": asr}


class TestRepeatedWords(unittest.TestCase):
    def test_words_a_corrected_line_repeats_are_dropped(self):
        script = [
            _line(1055.3, 1057.9, "While other kids were learning their ABCs.", asr="fixed"),
            _line(1058.4, 1061.4, "ABCs. She was learning how to KO the competition."),
        ]
        DubbingPipeline._trim_repeated_words(script)
        self.assertEqual(script[1]["text_en"], "She was learning how to KO the competition.")

    def test_lines_the_corrector_did_not_touch_keep_their_repetitions(self):
        script = [_line(0, 1, "We're done.", asr="sure"), _line(1.2, 2, "We're done. We're done.")]
        DubbingPipeline._trim_repeated_words(script)
        self.assertEqual(script[1]["text_en"], "We're done. We're done.")

    def test_a_whole_repeated_line_or_another_speaker_stays(self):
        script = [
            _line(0, 1, "Yeah!", asr="fixed"),
            _line(1.1, 2, "Yeah!"),
            _line(2.1, 3, "Yeah! Go!", speaker="S6"),
        ]
        DubbingPipeline._trim_repeated_words(script)
        self.assertEqual([s["text_en"] for s in script], ["Yeah!", "Yeah!", "Yeah! Go!"])


class TestSecondHearing(unittest.TestCase):
    def test_known_hallucinations_are_left_out(self):
        mgr = WhisperManager(inference_lock=None)
        mgr.model = MagicMock()
        segs = [
            SimpleNamespace(text=" Give me back.", no_speech_prob=0.1),
            SimpleNamespace(text=" We'll be right back.", no_speech_prob=0.1),
        ]
        mgr.model.transcribe.return_value = (iter(segs), None)
        with patch("core.whisper_engine.MOCK_MODE", False):
            self.assertEqual(mgr.transcribe_samples([0.0] * 16000, language="en"), "Give me back.")


class TestDoubtedName(unittest.TestCase):
    def _pipe(self):
        pipe = DubbingPipeline.__new__(DubbingPipeline)
        pipe.target_langs = ["pl"]
        pipe.media_info = MEDIA
        pipe.speaker_info = {"S3": {"name": "Chris Rose", "gender": "male"}}
        return pipe

    def _check(self, pipe, script):
        with (
            patch("core.pipeline.MOCK_MODE", False),
            patch("core.pipeline.speaker_pitch", return_value=255.0),
        ):
            pipe._check_genders({"S3": ["m.wav"]}, script)

    def test_the_cast_keeps_a_male_name_nobody_doubts(self):
        pipe = self._pipe()
        self._check(pipe, [_line(0, 1, "Hello.", speaker="S3")])
        self.assertEqual(pipe.speaker_info["S3"]["gender"], "male")

    def test_the_cast_fixes_the_analysis_gender_of_a_name(self):
        pipe = self._pipe()
        pipe.speaker_info["S3"]["gender"] = "female"
        self._check(pipe, [_line(0, 1, "Hello.", speaker="S3")])
        self.assertEqual(pipe.speaker_info["S3"]["gender"], "male")
        self.assertNotIn("gender_measured", pipe.speaker_info["S3"])

    def test_a_doubted_name_leaves_the_gender_alone(self):
        pipe = self._pipe()
        pipe.speaker_info["S3"]["gender"] = "female"
        line = _line(0, 1, "Let's go to my partners, the broadcasting machine Chris Rose.", speaker="S3")
        line["speaker_doubt"] = "Chris Rose is referring to himself in the third person"
        self._check(pipe, [line])
        self.assertEqual(pipe.speaker_info["S3"]["gender"], "female")


if __name__ == "__main__":
    unittest.main()
