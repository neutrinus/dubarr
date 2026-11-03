import os
import tempfile
import unittest
from unittest.mock import patch

from core import media_context
from core.media_context import format_media_context, format_short_context, lookup_media_info, parse_media_path
from infrastructure import metadata_client

VOX = "/tv/The Legend of Vox Machina/Season 2/The Legend of Vox Machina - S02E02 - The Trials of Vasselheim WEBDL-1080p.mkv"
SNW = "/tv/Star Trek - Strange New Worlds/Season 3/Star Trek - Strange New Worlds - S03E03 - Shuttle to Kenfori WEBDL-1080p Proper.mkv"


def fake_tvmaze(url, params=None, headers=None, timeout=10):
    if url.endswith("/search/shows"):
        return [
            {"show": {"id": 1, "name": "Star Trek: Strange New Worlds", "premiered": "2022-05-05"}},
            {"show": {"id": 2, "name": "Star Trek", "premiered": "1966-09-08"}},
        ]
    if url.endswith("/shows/1/cast"):
        return [
            {"person": {"name": "Anson Mount", "gender": "Male"}, "character": {"name": "Christopher Pike"}},
            {"person": {"name": "Rebecca Romijn", "gender": "Female"}, "character": {"name": "Una Chin-Riley"}},
        ]
    if url.endswith("/shows/1/episodebynumber"):
        assert params == {"season": 3, "number": 3}
        return {"id": 77, "name": "Shuttle to Kenfori", "summary": "<p>Pike &amp; crew crash.</p>"}
    if url.endswith("/episodes/77/guestcast"):
        return [{"person": {"name": "Guest Actor", "gender": "Female"}, "character": {"name": "Guest Hero"}}]
    return None


class TestParseMediaPath(unittest.TestCase):
    def test_sonarr_names(self):
        self.assertEqual(
            parse_media_path(VOX),
            {
                "kind": "episode",
                "title": "The Legend of Vox Machina",
                "year": None,
                "season": 2,
                "episode": 2,
                "episode_title": "The Trials of Vasselheim",
            },
        )
        info = parse_media_path(SNW)
        self.assertEqual((info["title"], info["season"], info["episode"]), ("Star Trek - Strange New Worlds", 3, 3))

    def test_folder_fallback_year_and_ids(self):
        info = parse_media_path("/tv/Doctor Who (2005) {tvdb-78804}/Season 01/S01E03.mkv")
        self.assertEqual((info["title"], info["year"], info["tvdb_id"]), ("Doctor Who", 2005, "78804"))

    def test_scene_and_movie_names(self):
        self.assertEqual(parse_media_path("/t/show.name.1x05.hdtv.mkv")["title"], "show name")
        movie = parse_media_path("/m/Dune.Part.Two.2024.2160p.WEB-DL.mkv")
        self.assertEqual((movie["kind"], movie["title"], movie["year"]), ("movie", "Dune Part Two", 2024))
        self.assertEqual(parse_media_path("/m/Inception (2010)/Inception (2010) Bluray-1080p.mkv")["title"], "Inception")
        self.assertEqual(parse_media_path("/m/Alien/Alien.1080p.BluRay.x264.mkv")["title"], "Alien")


class TestEpisodeNames(unittest.TestCase):
    def test_names_from_the_episode_title(self):
        path = (
            "/tv/BattleBots/Season 1/BattleBots - S01E09 - Alpha Raptor vs Mouser Mecha Catbot, Backlash vs Das Bot, "
            "Backlash vs Alpha Raptor (Lightweight Finals) WEBDL-480p.mkv"
        )
        info = parse_media_path(path)
        self.assertTrue(info["episode_title"].startswith("Alpha Raptor vs Mouser"))
        self.assertEqual(media_context.episode_names(info), ["Alpha Raptor", "Mouser Mecha Catbot", "Backlash", "Das Bot"])

    def test_plain_title_and_cast(self):
        info = {"episode_title": "Shuttle to Kenfori", "cast": [{"character": "Christopher Pike", "actor": "Anson Mount"}]}
        self.assertEqual(media_context.episode_names(info), ["Christopher Pike", "Anson Mount"])

    def test_scene_name_has_no_episode_title(self):
        self.assertNotIn("episode_title", parse_media_path("/t/show.name.1x05.hdtv.mkv"))


class TestLookup(unittest.TestCase):
    @patch.object(metadata_client, "_get_json", side_effect=fake_tvmaze)
    def test_tvmaze_episode(self, _):
        info = lookup_media_info(SNW)
        self.assertEqual(info["title"], "Star Trek: Strange New Worlds")
        self.assertEqual(info["episode_title"], "Shuttle to Kenfori")
        self.assertEqual(info["episode_plot"], "Pike & crew crash.")
        self.assertEqual(info["cast"][1], {"actor": "Rebecca Romijn", "character": "Una Chin-Riley", "gender": "female"})
        self.assertEqual(info["guest_cast"][0]["character"], "Guest Hero")
        self.assertEqual(info["sources"], ["tvmaze"])

        text = format_media_context(info)
        self.assertIn('Episode: S03E03 "Shuttle to Kenfori"', text)
        self.assertIn("- Christopher Pike (played by Anson Mount), male", text)
        self.assertIn("Characters in this episode:", text)
        self.assertEqual(format_short_context(info), 'Star Trek: Strange New Worlds (2022), S03E03 "Shuttle to Kenfori"')

    @patch.object(metadata_client, "_get_json", return_value=None)
    def test_nothing_found_keeps_file_name_guess(self, _):
        info = lookup_media_info(VOX)
        self.assertEqual(info["title"], "The Legend of Vox Machina")
        self.assertEqual(info["sources"], [])
        self.assertIn("TV series: The Legend of Vox Machina", format_media_context(info))

    @patch.object(metadata_client, "_get_json", side_effect=AssertionError("no network"))
    def test_offline_and_errors_never_raise(self, _):
        self.assertEqual(lookup_media_info(VOX, online=False)["title"], "The Legend of Vox Machina")
        with patch.object(media_context, "parse_media_path", side_effect=ValueError("boom")):
            self.assertEqual(lookup_media_info(VOX), {})
        self.assertEqual(format_media_context({}), "")

    @patch.object(metadata_client, "_get_json", side_effect=fake_tvmaze)
    def test_nfo_wins_and_gets_gender_from_tvmaze(self, _):
        with tempfile.TemporaryDirectory() as d:
            season = os.path.join(d, "Star Trek - Strange New Worlds", "Season 3")
            os.makedirs(season)
            video = os.path.join(season, os.path.basename(SNW))
            with open(os.path.join(os.path.dirname(season), "tvshow.nfo"), "w") as f:
                f.write(
                    "<tvshow><title>Star Trek: Strange New Worlds</title><plot>Space.</plot><genre>Sci-Fi</genre>"
                    "<actor><name>Anson Mount</name><role>Capt. Pike</role></actor></tvshow>\nhttps://example/1"
                )
            with open(os.path.splitext(video)[0] + ".nfo", "w") as f:
                f.write("<episodedetails><title>Shuttle</title><season>3</season><episode>3</episode></episodedetails>")
            info = lookup_media_info(video)
        self.assertEqual(info["plot"], "Space.")
        self.assertEqual(info["episode_title"], "Shuttle")
        self.assertEqual(info["cast"], [{"actor": "Anson Mount", "character": "Capt. Pike", "gender": "male"}])
        self.assertEqual(info["sources"], ["nfo", "tvmaze"])


class TestTmdb(unittest.TestCase):
    def test_movie_needs_key(self):
        self.assertIsNone(metadata_client.tmdb_movie_info("", "Inception", 2010))

    def test_movie(self):
        def fake(url, params=None, headers=None, timeout=10):
            if url.endswith("/search/movie"):
                assert params["year"] == 2010 and params["api_key"] == "k"
                return {"results": [{"id": 27205}]}
            if url.endswith("/movie/27205"):
                return {
                    "title": "Inception",
                    "release_date": "2010-07-15",
                    "overview": "Dreams.",
                    "genres": [{"name": "Action"}],
                    "credits": {"cast": [{"name": "Elliot Page", "character": "Ariadne", "gender": 3}]},
                }

        with patch.object(metadata_client, "_get_json", side_effect=fake):
            info = lookup_media_info("/m/Inception (2010)/Inception (2010) Bluray-1080p.mkv", tmdb_api_key="k")
        self.assertEqual((info["kind"], info["title"], info["plot"]), ("movie", "Inception", "Dreams."))
        self.assertEqual(info["cast"][0]["gender"], "non-binary")
