import os
import shutil
import tempfile
import unittest

from core.subtitles import (
    align,
    find_sidecar,
    grammar_gender,
    label_hints,
    lang_codes,
    missing_speech,
    parse_srt,
    read_text,
    shift_srt,
    speaker_labels,
    strip_sdh,
    subtitle_offset,
    timing_matches,
)

SRT = """1
00:00:01,000 --> 00:00:03,000
<i>Kapitanie, jesteśmy</i>
na orbicie.

2
00:00:04,500 --> 00:00:05,500
Dobrze.

3
00:00:20,000 --> 00:00:22,000
Nikt tego nie słyszał.
"""


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.video = os.path.join(self.dir, "Star Trek - S01E01 - Beyond.mkv")

    def touch(self, suffix, data=b"1"):
        path = os.path.splitext(self.video)[0] + suffix
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_target_and_source_sidecars(self):
        en = self.touch(".en.srt")
        hi = self.touch(".en.hi.srt")
        pl = self.touch(".pl.srt")
        self.touch(".pl.forced.srt")
        self.assertEqual(find_sidecar(self.video, langs=["pl"]), pl)
        self.assertEqual(find_sidecar(self.video, exclude=["pl"]), en)
        self.assertEqual(find_sidecar(self.video, exclude=["pl"], sdh=True), hi)
        self.assertIsNone(find_sidecar(self.video, langs=["de"]))

    def test_three_letter_codes(self):
        pol = self.touch(".pol.srt")
        self.assertEqual(find_sidecar(self.video, langs=["pl"]), pol)
        self.assertIsNone(find_sidecar(self.video, exclude=["pl"]))
        self.assertIn("ger", lang_codes("de"))

    def test_polish_subtitles_in_windows_1250(self):
        path = self.touch(".pl.srt", "Zażółć gęślą jaźń".encode("cp1250"))
        self.assertEqual(read_text(path), "Zażółć gęślą jaźń")
        path = self.touch(".en.srt", "﻿Hello".encode("utf-8"))
        self.assertEqual(read_text(path), "Hello")


class TestCues(unittest.TestCase):
    def test_parse(self):
        cues = parse_srt(SRT.replace("\n", "\r\n"))
        self.assertEqual(cues[0], (1.0, 3.0, "Kapitanie, jesteśmy na orbicie."))
        self.assertEqual(len(cues), 3)

    def test_each_cue_goes_to_the_line_it_overlaps_most(self):
        script = [
            {"index": 0, "start": 0.8, "end": 3.2, "speaker": "A"},
            {"index": 1, "start": 3.3, "end": 6.0, "speaker": "B"},
        ]
        aligned = align(script, parse_srt(SRT))
        self.assertEqual(aligned, {0: "Kapitanie, jesteśmy na orbicie.", 1: "Dobrze."})

    def test_sdh_labels_and_sounds(self):
        self.assertEqual(strip_sdh("KIRK: Mr. Spock, report. [beeping]"), (["Kirk"], "Mr. Spock, report."))
        self.assertEqual(strip_sdh("- SPOCK: Aye. - KIRK: Go."), (["Spock", "Kirk"], "Aye. Go."))
        self.assertEqual(strip_sdh("♪ theme music ♪"), ([], ""))
        self.assertEqual(strip_sdh("[MCCOY] He's dead, Jim."), (["Mccoy"], "He's dead, Jim."))


class TestMissingSpeech(unittest.TestCase):
    def cues(self, shift=0.0):
        return [(i * 3.0 + shift, i * 3.0 + 2.0 + shift, f"Line {i}.") for i in range(12)] + [
            (50.0, 52.0, "[explosion]"),
            (60.0, 63.0, "Whisper missed this."),
        ]

    def test_cues_that_whisper_did_not_hear(self):
        segments = [{"start": i * 3.0, "end": i * 3.0 + 2.0, "text": f"Line {i}."} for i in range(12)]
        self.assertTrue(timing_matches(segments, self.cues()))
        missing = missing_speech(segments, self.cues())
        self.assertEqual([(m["start"], m["text"]) for m in missing], [(60.0, "Whisper missed this.")])
        self.assertTrue(missing[0]["from_subtitles"])

    def test_subtitles_of_another_release_are_not_trusted(self):
        # Lines with pauses between them; the subtitles are 3 s late, so they fall into the pauses
        segments = [{"start": i * 6.0, "end": i * 6.0 + 2.0, "text": f"Line {i}."} for i in range(12)]
        cues = [(s + 3.0, s + 5.0, f"Line {i}.") for i, s in enumerate(range(0, 72, 6))]
        self.assertFalse(timing_matches(segments, cues))

    def test_a_cue_whisper_heard_at_another_time_is_not_added_again(self):
        # MythBusters: the subtitles ran 1-2 s early; "Alton, I bet you're kind of a badass" was dubbed twice
        segments = [
            {"start": 1715.3, "end": 1719.0, "text": "Alton, I'd bet you're kind of a bad guy when it comes to cooking."}
        ]
        cues = [(1713.0, 1715.0, "Alton, I bet you're kind of a badass"), (1725.0, 1727.0, "Yay! You did it!")]
        self.assertEqual([m["text"] for m in missing_speech(segments, cues)], ["Yay! You did it!"])
        # A short cue needs all its words nearby
        self.assertEqual(len(missing_speech([{"start": 10.0, "end": 12.0, "text": "Wow, look."}], [(7.0, 9.0, "Wow!")])), 0)
        self.assertEqual(len(missing_speech([{"start": 10.0, "end": 12.0, "text": "Look."}], [(7.0, 9.0, "Not yet.")])), 1)


class TestSubtitleOffset(unittest.TestCase):
    def test_offset_from_cues_that_start_like_a_segment(self):
        segments = [{"start": i * 6.0 + 1.5, "end": i * 6.0 + 4.0, "text": f"This is line number {i}."} for i in range(12)]
        cues = [(i * 6.0, i * 6.0 + 2.5, f"This is line number {i}.") for i in range(12)]
        self.assertEqual(subtitle_offset(segments, cues), 1.5)
        self.assertEqual(subtitle_offset(segments[:5], cues), 0.0)  # too few matches to tell
        self.assertEqual(subtitle_offset(segments, [(s + 1.4, e + 1.4, t) for s, e, t in cues]), 0.0)  # in time

    def test_shift_srt(self):
        shifted = shift_srt(SRT, 1.5)
        self.assertEqual(parse_srt(shifted)[0][:2], (2.5, 4.5))
        self.assertEqual(parse_srt(shift_srt(SRT, -5.0))[0][:2], (0.0, 0.0))
        self.assertEqual(shift_srt(SRT, 0.0), SRT)


class TestSpeakerLabels(unittest.TestCase):
    def test_label_of_most_cues(self):
        script = [
            {"index": 0, "start": 0.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"index": 1, "start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
            {"index": 2, "start": 10.0, "end": 12.0, "speaker": "SPEAKER_02"},
        ]
        cues = [(0.5, 2.0, "KIRK: Report."), (5.5, 7.0, "KIRK: Go."), (10.2, 11.0, "SPOCK: Aye.")]
        labels = speaker_labels(script, cues)
        self.assertEqual(labels, {"SPEAKER_01": {"Kirk": 2}, "SPEAKER_02": {"Spock": 1}})
        hints = label_hints(labels)
        self.assertIn("SPEAKER_01 is labelled Kirk in 2 of 2", hints)
        self.assertNotIn("SPEAKER_02", hints)  # one cue is not enough


class TestGrammarGender(unittest.TestCase):
    def test_first_person_forms(self):
        self.assertEqual(grammar_gender(["Byłem tam.", "Zrobiłem to.", "Przyszedłem."], "pl"), "male")
        self.assertEqual(grammar_gender(["Byłam tam.", "Zrobiłabym to.", "Widziałam go."], "pl"), "female")

    def test_nouns_and_dialogue_do_not_count(self):
        # "admirałem", "kołem" are nouns; a dialogue cue holds two speakers
        self.assertIsNone(grammar_gender(["Z admirałem.", "Kołem.", "Byłem."], "pl"))
        self.assertIsNone(grammar_gender(["- Byłem. - A ja byłam.", "- Zrobiłem. - Ja też zrobiłam."], "pl"))
        self.assertIsNone(grammar_gender(["Byłem.", "Byłam.", "Zrobiłem.", "Zrobiłam."], "pl"))  # mixed
        self.assertIsNone(grammar_gender(["Byłem.", "Zrobiłem.", "Widziałem."], "de"))


if __name__ == "__main__":
    unittest.main()
