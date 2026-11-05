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

    def test_each_task_gets_its_own_workspace(self):
        base = tempfile.mkdtemp()
        self.pipeline.base_temp_dir = base

        self.pipeline._enter_workspace(7)
        ws = self.pipeline.temp_dir
        self.assertEqual(ws, os.path.join(base, "task_7"))
        self.assertTrue(os.path.isdir(ws))
        self.assertEqual(self.pipeline.synchronizer.temp_dir, ws)
        self.assertEqual(self.mock_tts.temp_dir, ws)

        self.pipeline._leave_workspace()
        self.assertFalse(os.path.exists(ws))
        self.assertTrue(os.path.isdir(base))
        shutil.rmtree(base)

    def test_workspace_is_kept_in_debug_mode(self):
        base = tempfile.mkdtemp()
        self.pipeline.base_temp_dir = base
        self.pipeline.debug_mode = True

        self.pipeline._enter_workspace(None)
        self.pipeline._leave_workspace()

        self.assertTrue(os.path.isdir(os.path.join(base, "standalone")))
        shutil.rmtree(base)

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
        self.assertEqual(len(refs["A"]), 1)
        self.assertTrue(refs["A"][0].endswith("ref_A.wav"))
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
        # The noisy line stays as a lower candidate for the voice audition
        with open(os.path.join(self.pipeline.temp_dir, "ref_A_2.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), script[0]["text_en"])
        shutil.rmtree(self.pipeline.temp_dir)

    def test_extract_refs_keeps_up_to_the_audition_candidates(self):
        script = [
            {"speaker": "A", "start": 10.0 + 10 * n, "end": 18.0 + 10 * n, "text_en": f"Line number {n} of a calm talk here"}
            for n in range(5)
        ]
        self.pipeline.temp_dir = tempfile.mkdtemp()
        with (
            patch("core.pipeline.run_cmd"),
            patch("core.pipeline.measure_zcr", return_value=0.05),
            patch("core.pipeline.shutil.move"),
            patch("core.pipeline.REF_AUDITION_CANDIDATES", 3),
        ):
            refs = self.pipeline._extract_refs(script, "vocals.wav", "/tmp/ddir")
        self.assertEqual([os.path.basename(p) for p in refs["A"]], ["ref_A.wav", "ref_A_2.wav", "ref_A_3.wav"])
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


class TestVoiceAuditionStep(unittest.TestCase):
    def build(self, mode):
        with patch("core.pipeline.TAKE_CHECK", mode):
            return DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])

    def test_take_check_modes(self):
        judge, whisper, off = self.build("judge"), self.build("whisper"), self.build("off")
        self.assertIs(judge.synchronizer.judge, judge.judge)
        self.assertFalse(judge.judge.signal_only)
        self.assertIsNone(judge.synchronizer.take_checker)
        self.assertIsNotNone(whisper.synchronizer.take_checker)
        self.assertTrue(off.judge.signal_only)

    def test_audition_ranks_references_before_production(self):
        pipeline = self.build("judge")
        scores = {"A": {"a.wav": 0.9}}
        with patch("core.pipeline.MOCK_MODE", False), patch("core.pipeline.audition_voices", return_value=scores) as run:
            pipeline._audition_voices({"pl": [{"speaker": "A", "text": "x"}]})
        self.assertEqual(run.call_args.args[3], "pl")
        pipeline.tts_manager.apply_audition.assert_called_once_with(scores, 0.8)

    def test_no_audition_when_the_take_check_is_off(self):
        pipeline = self.build("off")
        with patch("core.pipeline.MOCK_MODE", False), patch("core.pipeline.audition_voices") as run:
            pipeline._audition_voices({"pl": []})
        run.assert_not_called()


class TestTakeCheckMode(unittest.TestCase):
    def test_values(self):
        from config import _take_check_mode

        self.assertEqual(_take_check_mode("judge"), "judge")
        self.assertEqual(_take_check_mode("on"), "judge")
        self.assertEqual(_take_check_mode("Audition"), "judge")
        self.assertEqual(_take_check_mode("whisper"), "whisper")
        self.assertEqual(_take_check_mode("off"), "off")
        self.assertEqual(_take_check_mode("false"), "off")


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


class TestRememberSeries(unittest.TestCase):
    def test_saves_the_used_names_and_never_fails(self):
        pipe = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])
        pipe.series = MagicMock()
        pipe.global_context = {"glossary": {"Razorback": "Rejzorbek", "Ice Wave": "Ajs Łejw"}}
        pipe._remember_series([{"text_en": "Razorback spins up."}])
        pipe.series.save_glossary.assert_called_once_with("pl", pipe.global_context["glossary"], ["Razorback"], replace=[])

        # The names' English phonemes are kept too, and a settled spelling that failed the check is replaced
        pipe.series.reset_mock()
        pipe.global_context = {
            "glossary": {"Ice Wave": "Ajs Łejw"},
            "name_phonemes": {"Ice Wave": "AY1 S W EY1 V"},
            "glossary_replaced": ["Ice Wave"],
        }
        pipe._remember_series([{"text_en": "Ice Wave wins."}])
        pipe.series.save_glossary.assert_any_call("pl", {"Ice Wave": "Ajs Łejw"}, ["Ice Wave"], replace=["Ice Wave"])
        pipe.series.save_glossary.assert_any_call("cmu", {"Ice Wave": "AY1 S W EY1 V"}, ["Ice Wave"])

        pipe.series.save_glossary.side_effect = OSError("disk full")
        pipe._remember_series([])  # logged, not raised


class TestCalmReferences(unittest.TestCase):
    def test_shouted_line_loses_to_calm_talk_after_it(self):
        pipe = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])
        lines = [{"start": float(t), "end": float(t) + 8} for t in (10, 30, 50, 70)]
        # Shouting at 10 s, the calm talk comes later in the episode
        levels = {10.0: -12.0, 30.0: -20.0, 50.0: -21.0, 70.0: -19.0}
        with patch("core.pipeline.voice_level", side_effect=lambda v, start, end: levels[start]):
            scored = pipe._apply_calm_scores([(s, 90) for s in lines], "vocals.wav", "A")
        best = max(scored, key=lambda x: x[1])[0]
        self.assertNotEqual(best["start"], 10.0)
        self.assertLess(dict((s["start"], v) for s, v in scored)[10.0], 90 - 20)

    def test_too_few_lines_leave_scores_alone(self):
        pipe = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])
        scored = [({"start": 1.0, "end": 5.0}, 50), ({"start": 9.0, "end": 14.0}, 60)]
        with patch("core.pipeline.voice_level", return_value=-20.0):
            self.assertEqual(pipe._apply_calm_scores(scored, "v.wav", "A"), scored)


class TestBankUpdate(unittest.TestCase):
    def test_speakers_with_three_lines_and_their_audited_real_clips(self):
        pipe = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])
        pipe.voiceprints = {"A": [1.0, 0.0]}
        pipe.voice_matches = {"A": "V001"}
        pipe.speaker_info = {"A": {"name": "Host", "gender": "male", "desc": "commentator"}}
        pipe.tts_manager.speaker_refs = {"A": ["ref_A.wav", "design_A.wav", "ref_A_2.wav"], "B": ["ref_B.wav"]}
        pipe.tts_manager.audition = {"ref_A.wav": 0.9, "design_A.wav": 0.95}
        pipe.tts_manager.designed = {"design_A.wav"}
        pipe.ref_energy = {"ref_A.wav": "calm"}
        pipe.shout_prints = {"A": [0.0, 1.0]}
        script = [{"speaker": "A"}] * 3 + [{"speaker": "B"}] * 2
        update = pipe._bank_update(script)
        self.assertEqual(list(update), ["A"])
        self.assertEqual(update["A"]["clips"], [("ref_A.wav", 0.9, None, "calm")])
        self.assertEqual(update["A"]["shout_print"], [0.0, 1.0])
        self.assertEqual((update["A"]["person"], update["A"]["profile"]["name"]), ("V001", "Host"))


class TestExtendShortClip(unittest.TestCase):
    def line(self, start, end, spk="A", text="x"):
        return {"start": start, "end": end, "speaker": spk, "text_en": text}

    def test_a_short_clip_grows_with_adjacent_lines_of_the_speaker(self):
        script = [
            self.line(0.0, 2.0, "B"),
            self.line(2.2, 4.0, text="Hello."),
            self.line(4.3, 7.0, text="Calm talk."),
            self.line(7.2, 15.0, text="Too long to join."),
        ]
        with patch("core.pipeline.clarity_db", return_value=12.0):
            out = DubbingPipeline._extend_short_clip(script[2], script, "v.wav")
        # The next line would make it 10.8 s; the previous one (another speaker before it) fits
        self.assertEqual((out["start"], out["end"], out["text_en"]), (2.2, 7.0, "Hello. Calm talk."))

    def test_noisy_or_distant_neighbours_do_not_join(self):
        script = [self.line(0.0, 2.0), self.line(3.0, 5.0), self.line(5.2, 7.0)]
        with patch("core.pipeline.clarity_db", return_value=1.0):  # crowd over the voice
            self.assertIs(DubbingPipeline._extend_short_clip(script[1], script, "v.wav"), script[1])
        long_clip = self.line(10.0, 17.0)
        self.assertIs(DubbingPipeline._extend_short_clip(long_clip, [long_clip], "v.wav"), long_clip)


class TestCheckGenders(unittest.TestCase):
    def test_pitch_never_changes_the_gender(self):
        # MythBusters: Jamie, Alton and Tory measured 220-237 Hz and spoke feminine Polish ("Próbowałam")
        pipe = DubbingPipeline.__new__(DubbingPipeline)
        pipe.target_langs = ["pl"]
        pipe.speaker_info = {"SPEAKER_05": {"name": "Alton Brown", "gender": "male"}}
        with (
            patch("core.pipeline.MOCK_MODE", False),
            patch("core.pipeline.speaker_pitch", return_value=225.0),
        ):
            pipe._check_genders({"SPEAKER_05": ["a.wav"]})
        self.assertEqual(pipe.speaker_info["SPEAKER_05"], {"name": "Alton Brown", "gender": "male"})


class TestAuditionScore(unittest.TestCase):
    def test_whisper_stands_in_for_a_missing_text_check(self):
        pipe = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])
        pipe.judge = MagicMock()
        pipe.judge.judge.return_value = {"score": 1.0, "problem": None, "voice": None, "text_match": None}
        pipe.synchronizer = MagicMock()
        pipe.synchronizer._intelligibility.return_value = (0.1, "Wawianca 5,3,3")
        self.assertEqual(pipe._audition_score("a.wav", "Po wielu długich godzinach w warsztacie", "pl", "ref.wav"), 0.1)

        pipe.judge.judge.return_value = {"score": 0.8, "problem": None, "voice": 0.7, "text_match": 0.8}
        self.assertEqual(pipe._audition_score("a.wav", "Po wielu długich godzinach w warsztacie", "pl", "ref.wav"), 0.8)


class TestSubtitlesInThePipeline(unittest.TestCase):
    def setUp(self):
        self.pipe = DubbingPipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(), target_langs=["pl"])
        self.dir = tempfile.mkdtemp()
        self.video = os.path.join(self.dir, "Show - S01E01.mkv")

    def write(self, suffix, text):
        with open(os.path.splitext(self.video)[0] + suffix, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def srt(cues):
        return "\n\n".join(f"{i + 1}\n00:00:{s:02d},000 --> 00:00:{e:02d},000\n{t}" for i, (s, e, t) in enumerate(cues))

    def test_target_cues_go_to_the_lines(self):
        self.write(".pl.srt", self.srt([(1, 3, "Byłem tam."), (5, 6, "Dobrze.")]))
        script = [
            {"index": 0, "start": 0.5, "end": 3.5, "speaker": "A"},
            {"index": 1, "start": 4.0, "end": 7.0, "speaker": "B"},
        ]
        self.pipe._attach_target_subtitles(self.video, script)
        self.assertEqual([s.get("subtitles") for s in script], [{"pl": "Byłem tam."}, {"pl": "Dobrze."}])

    def test_missing_speech_joins_the_transcription(self):
        cues = [(i * 4, i * 4 + 2, f"Line {i}.") for i in range(12)] + [(50, 53, "Unheard words.")]
        trans = [{"start": i * 4.0, "end": i * 4.0 + 2.0, "text": f"Line {i}."} for i in range(12)]
        out = self.pipe._add_missing_speech(trans, self.srt(cues))
        self.assertEqual(out[-1]["text"], "Unheard words.")
        self.assertEqual(len(out), 13)
        self.assertEqual(self.pipe._add_missing_speech(trans, ""), trans)

    def test_missing_speech_needs_somebody_speaking(self):
        cues = [(i * 4, i * 4 + 2, f"Line {i}.") for i in range(12)] + [(50, 53, "Unheard words."), (55, 58, "A sign.")]
        trans = [{"start": i * 4.0, "end": i * 4.0 + 2.0, "text": f"Line {i}."} for i in range(12)]
        diar = [{"start": 50.0, "end": 53.0, "speaker": "A"}]
        out = self.pipe._add_missing_speech(trans, self.srt(cues), diar)
        self.assertEqual([t["text"] for t in out[12:]], ["Unheard words."])

    def test_subtitle_credits_are_not_cues(self):
        from core.subtitles import is_credit, parse_srt

        for credit in (
            "MythBusters 11x07 Surreal Gourmet Hour Original air date Nov",
            "Sync and corrections by atrn97g for www.addic7ed.com",
            "Subtitles by explosiveskull",
            "Tłumaczenie: Jan Kowalski",
            "Downloaded from OpenSubtitles",
            "napisy24.pl",
        ):
            self.assertTrue(is_credit(credit), credit)
        for line in (
            "Let's sync the cameras.",
            "The translation is wrong, Spock.",
            "Napisy są złe.",
            "Mr. Spock, report.",
            "I lost 3.5 pounds.",
            "Corrections to the course, Mr. Sulu.",
        ):
            self.assertFalse(is_credit(line), line)
        cues = parse_srt(self.srt([(1, 3, "Sync and corrections by atrn97g for www.addic7ed.com"), (5, 6, "Hello.")]))
        self.assertEqual([c[2] for c in cues], ["Hello."])

    def test_grammar_gender_beats_the_analysis(self):
        self.pipe.speaker_info = {"SPEAKER_01": {"name": "Uhura", "gender": "female"}}
        script = [{"speaker": "SPEAKER_01", "subtitles": {"pl": t}} for t in ("Byłem tam.", "Zrobiłem to.", "Widziałem go.")]
        with (
            patch("core.pipeline.MOCK_MODE", False),
            patch("core.pipeline.speaker_pitch", return_value=240.0),
        ):
            self.pipe._check_genders({"SPEAKER_01": ["a.wav"]}, script)
        info = self.pipe.speaker_info["SPEAKER_01"]
        self.assertEqual((info["gender"], info["gender_measured"], info["name"]), ("male", True, ""))

    def test_terms_of_the_series_and_the_episode(self):
        self.pipe.series = MagicMock()
        self.pipe.series.terms.return_value = ({"Starfleet": "Gwiezdna Flota", "phaser": "fazer"}, {"A -> B": "ty"})
        self.pipe.global_context = {}
        self.pipe.llm_manager.extract_terms.return_value = {"terms": {"phaser": "fazer ręczny"}, "address": {}}
        script = [{"text_en": "Phaser!", "subtitles": {"pl": "Fazer!"}}]
        self.pipe._prepare_terms(script, "pl")
        self.assertEqual(self.pipe.global_context["terms"]["pl"], {"Starfleet": "Gwiezdna Flota", "phaser": "fazer ręczny"})
        self.assertEqual(self.pipe.global_context["address"]["pl"], {"A -> B": "ty"})
        # No subtitles: no LLM request, the series' terms only
        self.pipe.llm_manager.extract_terms.reset_mock()
        self.pipe._prepare_terms([{"text_en": "Hi."}], "pl")
        self.pipe.llm_manager.extract_terms.assert_not_called()
        self.pipe._remember_series([{"text_en": "Starfleet calls."}])
        self.pipe.series.save_terms.assert_called_with("pl", {}, {}, ["Starfleet"])
