import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from core.pipeline import DubbingPipeline


class TestDubbingPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_tts = MagicMock()
        self.mock_diar = MagicMock()
        self.mock_whisper = MagicMock()
        self.pipeline = DubbingPipeline(
            llm_manager=self.mock_llm,
            tts_manager=self.mock_tts,
            diar_manager=self.mock_diar,
            whisper_manager=self.mock_whisper,
            target_langs=["pl"],
            debug_mode=False,
        )

    def test_create_script(self):
        # Two practically continuous lines of the same speaker are merged
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 4.0}]
        trans = [{"start": 0.5, "end": 1.5, "text": "Hello"}, {"start": 1.6, "end": 3.5, "text": "World"}]

        script = self.pipeline._create_script(diar, trans)

        # Merged result: the pause is only 0.1 s
        self.assertEqual(len(script), 1)
        self.assertEqual(script[0]["text_en"], "Hello World")
        self.assertEqual(script[0]["speaker"], "SPEAKER_01")
        self.assertEqual(script[0]["start"], 0.5)
        self.assertEqual(script[0]["end"], 3.5)

    def test_create_script_does_not_merge_across_a_pause(self):
        # A 0.5 s pause stays split: speaking over it drifts out of sync with the original
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 4.0}]
        trans = [{"start": 0.5, "end": 2.0, "text": "Hello there."}, {"start": 2.5, "end": 4.0, "text": "World again."}]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual([s["text_en"] for s in script], ["Hello there.", "World again."])

    def test_create_script_joins_a_sentence_split_by_whisper(self):
        # The first part does not end a sentence: translating halves separately loses or duplicates content
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 9.0}]
        trans = [
            {"start": 0.0, "end": 3.0, "text": "His best defense is going to be a good"},
            {"start": 3.8, "end": 4.6, "text": "offense."},
            {"start": 5.5, "end": 8.0, "text": "That is a very destructive bot."},
        ]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual(
            [s["text_en"] for s in script],
            ["His best defense is going to be a good offense.", "That is a very destructive bot."],
        )
        self.assertEqual((script[0]["start"], script[0]["end"]), (0.0, 4.6))

    def test_create_script_never_joins_sentences_of_different_speakers(self):
        diar = [{"speaker": "A", "start": 0.0, "end": 3.0}, {"speaker": "B", "start": 3.0, "end": 6.0}]
        trans = [{"start": 0.0, "end": 2.5, "text": "So what do you"}, {"start": 3.2, "end": 5.5, "text": "think about it?"}]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual(len(script), 2)

    def test_create_script_clamps_a_line_stretched_over_music(self):
        diar = [{"speaker": "A", "start": 0.0, "end": 30.0}]
        trans = [{"start": 3.0, "end": 28.0, "text": "Ready for battle."}]

        script = self.pipeline._create_script(diar, trans)

        self.assertLess(script[0]["end"] - script[0]["start"], 5.0)
        self.assertEqual(script[0]["start"], 3.0)

    def test_create_script_clamps_a_short_sentence_stretched_over_fight_noise(self):
        # 13 syllables over 9.7 s: the dub would otherwise be expanded with padding to fill the slot
        diar = [{"speaker": "A", "start": 0.0, "end": 12.0}]
        trans = [{"start": 1.0, "end": 10.7, "text": "This bot of this match does some serious damage."}]

        script = self.pipeline._create_script(diar, trans)

        self.assertLess(script[0]["end"] - script[0]["start"], 6.0)

    def test_create_script_does_not_merge_beyond_the_duration_cap(self):
        # Back-to-back lines are still left separate when the block would get too long to stay in sync
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 16.0}]
        trans = [{"start": 0.0, "end": 9.0, "text": "First long part"}, {"start": 9.1, "end": 15.0, "text": "Second part"}]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual(len(script), 2)

    def test_create_script_merges_a_very_short_line_into_the_previous_one(self):
        # A 0.4 s slot can only hold a clipped fragment, so it is spoken together with the previous line
        diar = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 10.0}]
        trans = [
            {"start": 0.0, "end": 3.0, "text": "First line"},
            {"start": 4.0, "end": 4.4, "text": "O!"},
        ]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual(len(script), 1)
        self.assertEqual(script[0]["text_en"], "First line O!")
        self.assertEqual(script[0]["end"], 4.4)

    def test_create_script_merges_a_short_first_line_forward(self):
        diar = [{"speaker": "A", "start": 0.0, "end": 10.0}]
        trans = [{"start": 0.0, "end": 0.4, "text": "O!"}, {"start": 1.0, "end": 4.0, "text": "Real line"}]

        script = self.pipeline._create_script(diar, trans)

        self.assertEqual(len(script), 1)
        self.assertEqual(script[0]["text_en"], "O! Real line")
        self.assertEqual(script[0]["start"], 0.0)

    def test_create_script_different_speakers(self):
        diar = [{"speaker": "A", "start": 0.0, "end": 2.0}, {"speaker": "B", "start": 2.0, "end": 4.0}]
        trans = [{"start": 0.5, "end": 1.5, "text": "Hello"}, {"start": 2.5, "end": 3.5, "text": "World"}]
        script = self.pipeline._create_script(diar, trans)
        self.assertEqual(len(script), 2)
        self.assertEqual(script[0]["speaker"], "A")
        self.assertEqual(script[1]["speaker"], "B")

    def test_create_script_dominant_speaker(self):
        # Segment touches A briefly but is mostly spoken by B
        diar = [{"speaker": "A", "start": 0.0, "end": 1.2}, {"speaker": "B", "start": 1.2, "end": 5.0}]
        trans = [{"start": 1.0, "end": 4.0, "text": "Mostly B"}]
        script = self.pipeline._create_script(diar, trans)
        self.assertEqual(script[0]["speaker"], "B")

    def test_create_script_no_overlap(self):
        diar = [{"speaker": "A", "start": 10.0, "end": 12.0}]
        trans = [{"start": 0.0, "end": 1.0, "text": "Nobody"}]
        script = self.pipeline._create_script(diar, trans)
        self.assertEqual(script[0]["speaker"], "unknown")

    def test_create_script_slack_after(self):
        diar = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 2.0, "end": 7.55},
            {"speaker": "A", "start": 7.55, "end": 20.0},
        ]
        trans = [
            {"start": 0.0, "end": 3.0, "text": "This is line number one."},  # 1.0s gap to the next line -> 0.85s slack
            {"start": 4.0, "end": 6.0, "text": "This is line two."},  # 4.0s gap -> capped at 1s
            {"start": 10.0, "end": 12.0, "text": "This is line three."},  # 0.5s gap -> 0.35s slack
            {"start": 12.5, "end": 14.5, "text": "This is line four."},  # last segment -> capped
        ]
        with patch("core.pipeline.SLACK_AFTER_MAX", 1.0), patch("core.pipeline.SLACK_GUARD", 0.15):
            script = self.pipeline._create_script(diar, trans)
        self.assertEqual(len(script), 4)
        self.assertAlmostEqual(script[0]["max_end"], 3.85)
        self.assertAlmostEqual(script[1]["max_end"], 7.0)
        self.assertAlmostEqual(script[2]["max_end"], 12.35)
        self.assertAlmostEqual(script[3]["max_end"], 15.5)

    def test_cleanup_debug(self):
        with patch("shutil.rmtree") as mock_rm, patch("os.path.exists", return_value=True), patch("os.makedirs") as mock_mk:
            res = self.pipeline._cleanup_debug("test.mp4")
            mock_rm.assert_called_once()
            # Since debug_mode is False, it shouldn't call makedirs
            mock_mk.assert_not_called()
            self.assertTrue("debug_test" in res)

    def test_extract_refs_returns_golden_refs(self):
        script = [
            {"speaker": "A", "start": 10.0, "end": 18.0, "text_en": "This is a long and clean line of dialogue here"},
            {"speaker": "A", "start": 20.0, "end": 21.0, "text_en": "Short"},
        ]
        self.pipeline.speaker_refs = {"OLD": "/tmp/old.wav"}
        self.pipeline.temp_dir = tempfile.mkdtemp()
        with (
            patch("core.pipeline.run_cmd"),
            patch("core.pipeline.measure_zcr", return_value=0.05),
            patch("core.pipeline.shutil.move"),
        ):
            refs = self.pipeline._extract_refs(script, "vocals.wav", "/tmp/ddir")

        self.assertEqual(list(refs.keys()), ["A"])
        self.assertTrue(refs["A"].endswith("ref_A.wav"))
        self.assertIs(refs, self.pipeline.speaker_refs)
        with open(os.path.join(self.pipeline.temp_dir, "ref_A.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), script[0]["text_en"])
        shutil.rmtree(self.pipeline.temp_dir)

    def test_extract_refs_prefers_a_clear_line_over_a_noisy_one(self):
        script = [
            {"speaker": "A", "start": 10.0, "end": 18.0, "text_en": "Huge sparks with every hit from the spinner here"},
            {"speaker": "A", "start": 30.0, "end": 38.0, "text_en": "This next battle is going to be beautiful indeed"},
        ]
        self.pipeline.temp_dir = tempfile.mkdtemp()
        clarity = {10.0: 1.0, 30.0: 15.0}  # the first line is shouted over the crowd
        with (
            patch("core.pipeline.run_cmd"),
            patch("core.pipeline.measure_zcr", return_value=0.05),
            patch("core.pipeline.shutil.move"),
            patch("core.pipeline.clarity_db", side_effect=lambda v, start, end: clarity[start]),
        ):
            self.pipeline._extract_refs(script, "vocals.mp3", "/tmp/ddir")

        with open(os.path.join(self.pipeline.temp_dir, "ref_A.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), script[1]["text_en"])
        shutil.rmtree(self.pipeline.temp_dir)

    def test_extract_refs_keeps_the_speakers_own_voice_when_every_clip_is_noisy(self):
        script = [
            {"speaker": "A", "start": 10.0, "end": 18.0, "text_en": "Huge sparks with every hit from the spinner here"},
            {"speaker": "A", "start": 30.0, "end": 38.0, "text_en": "This next battle is going to be beautiful indeed"},
        ]
        self.pipeline.temp_dir = tempfile.mkdtemp()
        zcr = iter([0.3, 0.2])

        def cut(cmd, label):
            open(cmd[-2], "wb").close()

        with (
            patch("core.pipeline.run_cmd", side_effect=cut),
            patch("core.pipeline.measure_zcr", side_effect=lambda path: next(zcr)),
            patch("core.pipeline.shutil.move") as move,
            patch("core.pipeline.os.remove"),
        ):
            refs = self.pipeline._extract_refs(script, "vocals.mp3", "/tmp/ddir")

        self.assertIn("A", refs)
        # The second tested clip had the lower ZCR
        self.assertIn("test_A_30.0", move.call_args.args[0])
        shutil.rmtree(self.pipeline.temp_dir)


if __name__ == "__main__":
    unittest.main()


class TestSubtitleDiscovery(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.video = os.path.join(self.dir, "Show - S07E02 - The Well WEBDL-1080p.mkv")
        self.pipeline = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def touch(self, suffix):
        path = os.path.splitext(self.video)[0] + suffix
        open(path, "w").close()
        return path

    def test_prefers_full_english_over_hearing_impaired_and_skips_target_language(self):
        self.touch(".pl.srt")
        self.touch(".en.forced.srt")
        hi = self.touch(".en.hi.srt")
        self.assertEqual(self.pipeline._find_external_subtitles(self.video), hi)
        en = self.touch(".en.srt")
        self.assertEqual(self.pipeline._find_external_subtitles(self.video), en)
        plain = self.touch(".srt")
        self.assertEqual(self.pipeline._find_external_subtitles(self.video), plain)

    def test_only_target_language_subtitles(self):
        self.touch(".pl.srt")
        self.touch("-thumb.srt")
        self.assertIsNone(self.pipeline._find_external_subtitles(self.video))


class TestPauseBeforeLine(unittest.TestCase):
    def test_script_records_the_silence_before_each_line(self):
        pipeline = DubbingPipeline.__new__(DubbingPipeline)
        diar = [{"speaker": "A", "start": 0.0, "end": 3.0}, {"speaker": "B", "start": 3.0, "end": 9.0}]
        trans = [
            {"start": 1.0, "end": 3.0, "text": "This is the first line."},
            {"start": 3.3, "end": 5.0, "text": "This is the second one."},
            {"start": 7.0, "end": 9.0, "text": "And this is the third."},
        ]
        with patch("core.pipeline.SLACK_BEFORE_MAX", 0.5), patch("core.pipeline.SLACK_GUARD", 0.15):
            script = pipeline._create_script(diar, trans)
        self.assertAlmostEqual(script[0]["min_start"], 0.5)
        self.assertAlmostEqual(script[1]["min_start"], 3.15)  # only 0.3 s pause, minus the guard
        self.assertAlmostEqual(script[2]["min_start"], 6.5)

    def test_take_starts_early_only_when_it_would_overrun(self):
        item = {"start": 10.0, "end": 12.0, "max_end": 12.5, "min_start": 9.5}
        self.assertEqual(DubbingPipeline._place_take(item, 2.3, 0.0), (10.0, 1.0))
        start, speed = DubbingPipeline._place_take(item, 2.8, 0.0)
        self.assertAlmostEqual(start, 9.7)
        self.assertEqual(speed, 1.0)

    def test_early_start_never_overlaps_the_previous_take(self):
        item = {"start": 10.0, "end": 12.0, "max_end": 12.5, "min_start": 9.5}
        start, speed = DubbingPipeline._place_take(item, 3.0, 9.9)
        self.assertAlmostEqual(start, 9.9)
        self.assertAlmostEqual(speed, 3.0 / 2.6)  # what does not fit is sped up instead
