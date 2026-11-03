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
