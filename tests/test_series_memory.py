import json
import os
import tempfile
import unittest
from unittest.mock import patch

from core import series_memory
from core.series_memory import SeriesMemory, series_key


class TestSeriesKey(unittest.TestCase):
    def test_episode_title(self):
        self.assertEqual(series_key({"kind": "episode", "title": "BattleBots"}), "battlebots")
        self.assertEqual(
            series_key({"kind": "episode", "title": "Star Trek: Strange New Worlds"}), "star-trek-strange-new-worlds"
        )

    def test_no_memory_for_movies_or_unknown(self):
        self.assertIsNone(series_key({"kind": "movie", "title": "Dune"}))
        self.assertIsNone(series_key({}))
        self.assertIsNone(series_key({"kind": "episode", "title": "???"}))

    def test_switched_off(self):
        with patch.object(series_memory, "SERIES_MEMORY", False):
            self.assertIsNone(SeriesMemory.for_media({"kind": "episode", "title": "BattleBots"}))


class TestGlossary(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.mem = SeriesMemory.for_media({"kind": "episode", "title": "BattleBots"}, root=self.root)

    def test_round_trip_and_stored_spelling_is_never_replaced(self):
        self.assertEqual(self.mem.glossary("pl"), {})
        self.mem.save_glossary("pl", {"Razorback": "Rejzorbek", "Ice Wave": "Ajs Łejw"})
        self.mem.save_glossary("pl", {"Razorback": "Rej-or-bek", "Backlash": "Beklesz"}, used_terms=["Razorback"])
        self.assertEqual(self.mem.glossary("pl"), {"Razorback": "Rejzorbek", "Ice Wave": "Ajs Łejw", "Backlash": "Beklesz"})
        with open(os.path.join(self.root, "battlebots", "glossary.json")) as f:
            data = json.load(f)
        self.assertEqual(data["pl"]["Razorback"]["episodes"], 2)
        self.assertEqual(data["pl"]["Ice Wave"]["episodes"], 1)

    def test_languages_are_kept_apart(self):
        self.mem.save_glossary("pl", {"Razorback": "Rejzorbek"})
        self.assertEqual(self.mem.glossary("de"), {})

    def test_broken_file_starts_from_scratch(self):
        os.makedirs(self.mem.dir)
        with open(self.mem.glossary_path, "w") as f:
            f.write("{broken")
        self.assertEqual(self.mem.glossary("pl"), {})


if __name__ == "__main__":
    unittest.main()


class TestVoiceBank(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.mem = SeriesMemory("battlebots", root=self.root)
        self.work = tempfile.mkdtemp()

    def clip(self, name, text="Calm talk before the fight."):
        path = os.path.join(self.work, name)
        with open(path, "wb") as f:
            f.write(b"RIFF....WAVE")
        with open(os.path.splitext(path)[0] + ".txt", "w") as f:
            f.write(text)
        return path

    def first_episode(self):
        self.mem.save_voices(
            {
                "SPEAKER_00": {
                    "person": None,
                    "print": [1.0, 0.0, 0.0],
                    "profile": {"name": "Host", "gender": "male", "desc": "commentator"},
                    "clips": [(self.clip("ref_00.wav"), 0.9), (self.clip("ref_00_2.wav"), 0.4)],
                },
                "SPEAKER_01": {"person": None, "print": [0.0, 1.0, 0.0], "profile": {"gender": "unknown"}, "clips": []},
                "SPEAKER_02": {"person": None, "print": None, "profile": {}, "clips": []},  # no print: skipped
            }
        )

    def test_new_persons_and_their_clips(self):
        self.first_episode()
        persons = self.mem.persons()
        self.assertEqual(sorted(persons), ["V001", "V002"])
        self.assertEqual(persons["V001"]["name"], "Host")
        self.assertNotIn("gender", persons["V002"])  # "unknown" is not stored
        clips = self.mem.clips("V001")
        self.assertEqual(len(clips), 2)
        self.assertTrue(os.path.exists(os.path.splitext(clips[0])[0] + ".txt"))

    def test_a_measured_gender_replaces_a_guessed_one(self):
        self.first_episode()
        # A guess replaces a guess: the shouting man stored as a woman is not kept forever
        profile = {"name": "Jenny Taft", "gender": "female", "gender_measured": False}
        self.mem.save_voices({"SPEAKER_03": {"person": "V001", "print": None, "profile": profile, "clips": []}})
        self.assertEqual(self.mem.persons()["V001"]["gender"], "female")
        profile = {"gender": "male", "gender_measured": True}
        self.mem.save_voices({"SPEAKER_03": {"person": "V001", "print": None, "profile": profile, "clips": []}})
        person = self.mem.persons()["V001"]
        self.assertEqual(person["gender"], "male")
        self.assertNotIn("name", person)  # "Host" was stored under a woman's gender: someone else's name
        # A measured gender is not replaced by a guess
        profile = {"name": "Chris Rose", "gender": "female", "gender_measured": False}
        self.mem.save_voices({"SPEAKER_03": {"person": "V001", "print": None, "profile": profile, "clips": []}})
        person = self.mem.persons()["V001"]
        self.assertEqual((person["gender"], person["name"]), ("male", "Chris Rose"))

    def test_an_old_pitch_verdict_counts_as_a_guess(self):
        # Banks before 2026-11-05 stored pitch verdicts (gender_hz): a shouting man as a woman
        self.first_episode()
        persons = self.mem.persons()
        persons["V001"].update({"gender": "female", "gender_measured": True, "gender_hz": 293, "episodes": 3})
        series_memory._write_json(self.mem.voices_path, {"persons": persons})
        self.assertEqual(self.mem.settled_genders({"SPEAKER_00": "V001"}), {})
        self.assertNotIn("female", self.mem.hints({"SPEAKER_00": "V001"}))
        self.mem.save_voices({"SPEAKER_00": {"person": "V001", "print": None, "profile": {"gender": "male"}, "clips": []}})
        self.assertEqual(self.mem.persons()["V001"]["gender"], "male")

    def test_a_cleaner_clip_wins_between_equal_scores(self):
        self.first_episode()
        # Same audition score as the stored 0.9 clip, but longer and cleaner; the 0.4 one is pushed out first
        with patch.object(series_memory, "VOICE_BANK_CLIPS", 2):
            self.mem.save_voices(
                {"SPEAKER_00": {"person": "V001", "print": None, "profile": {}, "clips": [(self.clip("long.wav"), 0.88, 1.0)]}}
            )
        clips = self.mem.persons()["V001"]["clips"]
        self.assertEqual([c["score"] for c in clips], [0.88, 0.9])
        self.assertEqual(clips[0]["quality"], 1.0)

    def test_clip_quality(self):
        from core.series_memory import clip_quality

        self.assertEqual(clip_quality(8.0, 15.0), 1.0)
        self.assertLess(clip_quality(3.0, 15.0), clip_quality(7.0, 15.0))
        self.assertLess(clip_quality(8.0, 4.0), clip_quality(8.0, 12.0))
        self.assertEqual(clip_quality(8.0, None), 0.75)  # clarity unknown: middle

    def test_match_needs_a_clear_winner(self):
        self.first_episode()
        matches = self.mem.match({"SPEAKER_05": [0.95, 0.1, 0.0], "SPEAKER_06": [0.7, 0.7, 0.0], "X": [0.0, 0.0, 1.0]})
        # SPEAKER_06 is as close to both persons: not matched; X is nobody known
        self.assertEqual(matches, {"SPEAKER_05": "V001"})

    def test_hints_and_settled_gender(self):
        self.first_episode()
        # A guessed gender is neither repeated to the analysis nor settled
        self.assertIn("SPEAKER_03 is probably Host, commentator", self.mem.hints({"SPEAKER_03": "V001"}))
        self.mem.save_voices({"SPEAKER_03": {"person": "V001", "print": [0.9, 0.1, 0.0], "profile": {}, "clips": []}})
        self.assertEqual(self.mem.settled_genders({"SPEAKER_03": "V001"}), {})
        self.assertEqual(self.mem.persons()["V001"]["episodes"], 2)
        self.assertAlmostEqual(self.mem.persons()["V001"]["print"][0], 0.95)
        measured = {"gender": "male", "gender_measured": True, "gender_hz": 130}
        self.mem.save_voices({"SPEAKER_03": {"person": "V001", "print": None, "profile": measured, "clips": []}})
        self.assertIn("SPEAKER_03 is probably Host, male, commentator", self.mem.hints({"SPEAKER_03": "V001"}))
        self.assertEqual(self.mem.settled_genders({"SPEAKER_03": "V001"}), {"SPEAKER_03": "male"})

    def test_only_the_best_clips_are_kept(self):
        self.first_episode()
        with patch.object(series_memory, "VOICE_BANK_CLIPS", 2):
            self.mem.save_voices(
                {"S": {"person": "V001", "print": None, "profile": {}, "clips": [(self.clip("calm.wav"), 0.95)]}}
            )
        clips = self.mem.persons()["V001"]["clips"]
        self.assertEqual([c["score"] for c in clips], [0.95, 0.9])
        self.assertEqual(len(os.listdir(self.mem.clips_dir)), 4)  # two clips with their transcripts

    def test_merge_split_speakers(self):
        diar = [{"speaker": "SPEAKER_01", "start": 0, "end": 1}, {"speaker": "SPEAKER_02", "start": 1, "end": 2}]
        prints = {"SPEAKER_01": [1.0, 0.0], "SPEAKER_02": [0.8, 0.2], "SPEAKER_03": [0.0, 1.0]}
        matches = {"SPEAKER_01": "V001", "SPEAKER_02": "V001"}
        diar, prints, matches = series_memory.merge_matched_speakers(diar, prints, matches)
        self.assertEqual({d["speaker"] for d in diar}, {"SPEAKER_01"})
        self.assertEqual(matches, {"SPEAKER_01": "V001"})
        self.assertEqual(prints["SPEAKER_01"], [0.9, 0.1])
        self.assertIn("SPEAKER_03", prints)


def test_a_spelling_that_failed_the_check_is_replaced(tmp_path):
    from core.series_memory import SeriesMemory

    mem = SeriesMemory("battlebots", root=str(tmp_path))
    mem.save_glossary("pl", {"Warhead": "Worehed", "Razorback": "Rejzorbek"})
    mem.save_glossary("pl", {"Warhead": "Łorhed", "Razorback": "Rej-zor-bek"}, replace=["Warhead"])
    assert mem.glossary("pl") == {"Warhead": "Łorhed", "Razorback": "Rejzorbek"}


class TestTerms(unittest.TestCase):
    def test_terms_and_address_across_episodes(self):
        mem = SeriesMemory("star-trek", root=tempfile.mkdtemp())
        self.assertEqual(mem.terms("pl"), ({}, {}))
        mem.save_terms("pl", {"Starfleet": "Gwiezdna Flota", "phaser": "fazer"}, {"Spock -> Kirk": "formal"})
        mem.save_terms("pl", {"phaser": "fazer ręczny"}, {}, used_terms=["Starfleet"])
        terms, address = mem.terms("pl")
        self.assertEqual(terms, {"Starfleet": "Gwiezdna Flota", "phaser": "fazer ręczny"})  # the latest subtitles win
        self.assertEqual(address, {"Spock -> Kirk": "formal"})
        with open(mem.terms_path) as f:
            stored = json.load(f)["pl"]["terms"]
        self.assertEqual(stored["Starfleet"]["episodes"], 2)
        self.assertEqual(mem.terms("de"), ({}, {}))
