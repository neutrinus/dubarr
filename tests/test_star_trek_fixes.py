"""Fixes after the Star Trek TAS S01E01 run (gap-pass hallucinations, stretched segments, cast gender, glossary)."""

import unittest
from unittest.mock import patch

from core.media_context import cast_gender
from core.pipeline import DubbingPipeline
from utils import check_polish_spelling, strip_title

MEDIA = {
    "cast": [{"actor": "William Shatner", "character": "Captain James Tiberius Kirk", "gender": "male"}],
    "guest_cast": [{"actor": "Nichelle Nichols", "character": "Lt. Nyota Uhura", "gender": "female"}],
}


class TestGapHallucinations(unittest.TestCase):
    def test_gap_segment_without_speech_or_cue_is_dropped(self):
        trans = [
            {"start": 0.0, "end": 2.0, "text": "Space."},
            {"start": 61.0, "end": 62.0, "text": "We'll be right back.", "gap": True},
            {"start": 100.0, "end": 101.5, "text": "Obey me!", "gap": True},
            {"start": 200.0, "end": 201.0, "text": "Captain?", "gap": True},
        ]
        diar = [{"start": 99.9, "end": 101.6, "speaker": "S3"}]
        cues = [(199.8, 201.2, "Captain?")]
        kept = DubbingPipeline._drop_unheard_gap_segments(trans, diar, cues)
        self.assertEqual([t["text"] for t in kept], ["Space.", "Obey me!", "Captain?"])


class TestStretchedWindow(unittest.TestCase):
    def test_window_goes_where_the_speech_is(self):
        seg = {"start": 1223.98, "end": 1240.5, "text": "That's the heart of a galaxy, Captain."}
        diar = [{"start": 1238.9, "end": 1240.4, "speaker": "K"}]
        out = DubbingPipeline._clamp_stretched(seg, diar)
        self.assertLessEqual(out["start"], 1238.9)
        self.assertGreater(out["start"], 1234.0)
        self.assertLessEqual(out["end"], 1240.5)

    def test_without_speech_it_keeps_the_start(self):
        seg = {"start": 10.0, "end": 40.0, "text": "Ready for battle."}
        self.assertEqual(DubbingPipeline._clamp_stretched(seg, [])["start"], 10.0)
        short = {"start": 10.0, "end": 12.0, "text": "Ready for battle."}
        self.assertIs(DubbingPipeline._clamp_stretched(short, []), short)


class TestCastGender(unittest.TestCase):
    def test_lookup(self):
        self.assertEqual(cast_gender(MEDIA, "Captain James Tiberius Kirk"), "male")
        self.assertEqual(cast_gender(MEDIA, "Kirk"), "male")
        self.assertEqual(cast_gender(MEDIA, "Uhura"), "female")
        self.assertIsNone(cast_gender(MEDIA, "Narrator"))
        self.assertIsNone(cast_gender({}, "Kirk"))

    def test_the_cast_decides_and_the_pitch_does_not(self):
        pipe = DubbingPipeline.__new__(DubbingPipeline)
        pipe.target_langs = ["pl"]
        pipe.media_info = MEDIA
        pipe.speaker_info = {
            "S7": {"name": "Captain James Tiberius Kirk", "gender": "female"},
            "S0": {"name": "Lt. Nyota Uhura", "gender": "female"},
        }
        with (
            patch("core.pipeline.MOCK_MODE", False),
            patch("core.pipeline.speaker_pitch", return_value=244.0),
        ):
            pipe._check_genders({"S7": ["k.wav"], "S0": ["u.wav"]})
        self.assertEqual(pipe.speaker_info["S7"]["gender"], "male")
        self.assertEqual(pipe.speaker_info["S7"]["name"], "Captain James Tiberius Kirk")
        self.assertEqual((pipe.speaker_info["S0"]["gender"], pipe.speaker_info["S0"]["name"]), ("female", "Lt. Nyota Uhura"))


class TestGlossaryNames(unittest.TestCase):
    def test_titles_are_cut(self):
        self.assertEqual(
            strip_title("Captain James Tiberius Kirk", "Kapten Dżejms Tajberius Kark"),
            ("James Tiberius Kirk", "Dżejms Tajberius Kark"),
        )
        self.assertEqual(strip_title("Lt. Nyota Uhura", "Letnant Njota Uhura"), ("Nyota Uhura", "Njota Uhura"))
        self.assertEqual(strip_title("Dr. McCoy", "Makkoj"), ("McCoy", "McCoy"))  # no word to cut: as written
        self.assertEqual(strip_title("Captain", "Kapitan"), ("Captain", "Kapitan"))
        self.assertEqual(strip_title("Razorback", "Rejzorbek"), ("Razorback", "Rejzorbek"))

    def test_stressed_vowel_must_survive(self):
        self.assertFalse(check_polish_spelling("Kark", "K ER1 K")[0])
        self.assertTrue(check_polish_spelling("Kirk", "K ER1 K")[0])
        self.assertTrue(check_polish_spelling("Kerk", "K ER1 K")[0])
        self.assertTrue(check_polish_spelling("Spok", "S P AA1 K")[0])
        self.assertTrue(check_polish_spelling("Ajs Łejw", "AY1 S W EY1 V")[0])
        self.assertTrue(check_polish_spelling("Rejzorbek", "R EY1 Z ER0 B AE2 K")[0])


if __name__ == "__main__":
    unittest.main()
