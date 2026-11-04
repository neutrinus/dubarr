import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import script_check as sc
from core.llm_engine import LLMManager, apply_editor_flags
from core.pipeline import DubbingPipeline, ref_file_energy, ref_file_name, script_names
from core.series_memory import SeriesMemory
from core.tts_manager import TTSManager
from utils import count_syllables


def tone(seconds, amp=0.1):
    return (amp * np.sin(np.arange(int(seconds * sc.RATE)) * 0.05)).astype(np.float32)


class TestConfidence(unittest.TestCase):
    def test_unsure_by_logprob_or_word(self):
        self.assertFalse(sc.is_unsure({"avg_logprob": -0.2, "min_word_prob": 0.8}))
        self.assertTrue(sc.is_unsure({"avg_logprob": -0.9}))
        self.assertTrue(sc.is_unsure({"avg_logprob": -0.1, "min_word_prob": 0.1}))
        self.assertFalse(sc.is_unsure({}))

    def test_subtitle_text_needs_a_third_of_the_cue(self):
        cues = [(0.0, 3.0, "First."), (2.9, 6.0, "Second."), (10.0, 12.0, "Far.")]
        self.assertEqual(sc.subtitle_text(cues, 0.0, 3.0), "First.")
        self.assertEqual(sc.subtitle_text(cues, 0.0, 5.0), "First. Second.")
        self.assertEqual(sc.subtitle_text(None, 0.0, 5.0), "")


class TestSecondHearing(unittest.TestCase):
    def test_only_unsure_segments_are_heard_again(self):
        audio = tone(10)
        calls = []

        def transcribe(samples, prompt):
            calls.append(prompt)
            return "It's Robo-X!" if prompt else "It's Robux"

        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello there.", "avg_logprob": -0.1},
            {"start": 3.0, "end": 4.0, "text": "It's Robux!", "avg_logprob": -1.2},
        ]
        out = sc.second_hearing(segments, transcribe, audio, audio, ["Robo-X"], [(3.0, 4.0, "It's Robo-X!")])
        self.assertNotIn("unsure", out[0])
        self.assertTrue(out[1]["unsure"])
        # The vocals hearing equals the first one (no punctuation aside): kept, it differs as text
        self.assertEqual(out[1]["hearings"]["prompted"], "It's Robo-X!")
        self.assertEqual(out[1]["hearings"]["subtitle"], "It's Robo-X!")
        self.assertEqual(len(calls), 2)
        self.assertIn("Robo-X", calls[1])
        self.assertIn("Hello there.", calls[1])  # the previous line as context

    def test_same_hearing_is_not_kept(self):
        audio = tone(5)
        out = sc.second_hearing(
            [{"start": 0.0, "end": 1.0, "text": "Go!", "avg_logprob": -1.0}], lambda s, p: "Go!", audio, audio, []
        )
        self.assertTrue(out[0]["unsure"])
        self.assertNotIn("hearings", out[0])

    def test_line_hearings_keep_sure_parts(self):
        parts = [{"text": "Here comes"}, {"text": "Robux!", "hearings": {"prompted": "Robo-X!"}}]
        self.assertEqual(sc.line_hearings(parts), {"prompted": "Here comes Robo-X!"})
        self.assertEqual(sc.line_hearings([{"text": "Sure."}]), {})


class TestSpeakerRecheck(unittest.TestCase):
    def setUp(self):
        self.centroids = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
        self.vocals = tone(20)

    def test_clearly_closer_voice_moves_the_segment(self):
        segs = [{"start": 0.0, "end": 3.0, "speaker": "A", "text": "x"}]
        out = sc.recheck_speakers(segs, self.centroids, lambda s: np.array([0.1, 0.9]), self.vocals)
        self.assertEqual((out[0]["speaker"], out[0]["speaker_moved_from"]), ("B", "A"))

    def test_close_call_or_short_segment_stays(self):
        segs = [
            {"start": 0.0, "end": 3.0, "speaker": "A", "text": "x"},
            {"start": 5.0, "end": 5.5, "speaker": "A", "text": "y"},
        ]
        out = sc.recheck_speakers(segs, self.centroids, lambda s: np.array([0.68, 0.72]), self.vocals)
        self.assertEqual([s["speaker"] for s in out], ["A", "A"])
        out = sc.recheck_speakers(segs[1:], self.centroids, lambda s: np.array([0.0, 1.0]), self.vocals)
        self.assertEqual(out[0]["speaker"], "A")  # too short for a voiceprint

    def test_weak_match_stays(self):
        segs = [{"start": 0.0, "end": 3.0, "speaker": "A", "text": "x"}]
        out = sc.recheck_speakers(segs, self.centroids, lambda s: np.array([-0.5, 0.4]), self.vocals)
        self.assertEqual(out[0]["speaker"], "A")


class TestEnergy(unittest.TestCase):
    def test_level_is_of_the_voice_not_the_pauses(self):
        quiet = np.concatenate([tone(1, 0.1), np.zeros(sc.RATE, dtype=np.float32)])
        self.assertAlmostEqual(sc.level_db(quiet), sc.level_db(tone(1, 0.1)), delta=1.0)
        self.assertGreater(sc.level_db(tone(1, 0.4)), sc.level_db(tone(1, 0.1)) + 10)
        self.assertIsNone(sc.level_db(None))

    def test_classes_relative_to_the_speaker(self):
        med = {"level": -30.0, "pitch": 120.0, "pace": 4.0}
        self.assertEqual(sc.energy_class({"level": -30.0, "pitch": 120.0, "pace": 4.0}, med), "calm")
        self.assertEqual(sc.energy_class({"level": -27.0, "pitch": 120.0, "pace": 4.0}, med), "lively")
        self.assertEqual(sc.energy_class({"level": -24.0, "pitch": 160.0, "pace": 4.0}, med), "shout")
        self.assertEqual(sc.energy_class({"level": -28.5, "pitch": 120.0, "pace": 6.0}, med), "lively")  # fast
        self.assertIsNone(sc.energy_class({"level": None}, med))

    def test_assign_energy(self):
        script = [{"index": i, "speaker": "A"} for i in range(5)]
        measures = {i: {"level": -30.0, "pitch": 120.0, "pace": 4.0} for i in range(4)}
        measures[4] = {"level": -20.0, "pitch": 200.0, "pace": 4.0}
        counts = sc.assign_energy(script, measures)
        self.assertEqual(script[4]["energy"], "shout")
        self.assertEqual(counts["A"], {"calm": 4, "shout": 1})

    def test_class_distance(self):
        self.assertEqual(sc.class_distance("calm", "shout"), 2)
        self.assertEqual(sc.class_distance("lively", "lively"), 0)
        self.assertEqual(sc.class_distance(None, "calm"), 1)


class TestShoutMerge(unittest.TestCase):
    def setUp(self):
        # SPEAKER_04 shouting, SPEAKER_13 the same man calm (S01E03)
        self.script = [{"index": i, "speaker": "SPEAKER_04"} for i in range(3)]
        self.script += [{"index": i, "speaker": "SPEAKER_13"} for i in range(3, 6)]
        self.measures = {i: {"level": -18.0, "pitch": 300.0, "pace": 5.0} for i in range(3)}
        self.measures.update({i: {"level": -26.0, "pitch": 140.0, "pace": 4.0} for i in range(3, 6)})
        self.info = {"SPEAKER_04": {"name": "Chris Rose"}, "SPEAKER_13": {"name": "Chris Rose"}}
        self.prints = {"SPEAKER_04": [1.0, 0.3], "SPEAKER_13": [1.0, 0.0]}

    def test_merged_when_all_signals_agree(self):
        merges = sc.shout_merges(self.script, self.measures, self.info, self.prints, {})
        self.assertEqual(merges, {"SPEAKER_04": "SPEAKER_13"})

    def test_name_alone_is_not_enough(self):
        prints = {"SPEAKER_04": [0.0, 1.0], "SPEAKER_13": [1.0, 0.0]}
        self.assertEqual(sc.shout_merges(self.script, self.measures, self.info, prints, {}), {})
        genders = {"SPEAKER_04": "female", "SPEAKER_13": "male"}
        self.assertEqual(sc.shout_merges(self.script, self.measures, self.info, self.prints, genders), {})
        info = {"SPEAKER_04": {"name": "Jenny Taft"}, "SPEAKER_13": {"name": "Chris Rose"}}
        self.assertEqual(sc.shout_merges(self.script, self.measures, info, self.prints, {}), {})
        info = {"SPEAKER_04": {"name": "Unknown"}, "SPEAKER_13": {"name": "Unknown"}}
        self.assertEqual(sc.shout_merges(self.script, self.measures, info, self.prints, {}), {})

    def test_not_louder_is_not_merged(self):
        for i in range(3):
            self.measures[i]["level"] = -25.0
        self.assertEqual(sc.shout_merges(self.script, self.measures, self.info, self.prints, {}), {})


class TestEditorAnswer(unittest.TestCase):
    def setUp(self):
        self.script = [
            {"text_en": "It's Robux!", "asr_unsure": True},
            {"text_en": "This is a long and perfectly confident sentence.", "speaker": "A"},
            {"text_en": "Thank you.", "speaker": "A"},
        ]

    def test_structured_answer(self):
        answer = {
            "fixes": {"L_0": "It's Robo-X!", "7": "out of chunk"},
            "guessed": ["L_0"],
            "hallucinations": ["L_1", "L_2"],
            "speaker_doubts": {"L_2": "Thanks the host while being the host"},
        }
        fixes, flags, doubts = LLMManager._editor_answer(answer, self.script, 0, 3)
        self.assertEqual(fixes, {0: "It's Robo-X!"})
        # A long confident line is speech, not a hallucination
        self.assertEqual(flags, {0: "guessed", 2: "hallucination"})
        self.assertEqual(list(doubts), [2])

    def test_old_flat_answer_is_fixes(self):
        fixes, flags, doubts = LLMManager._editor_answer({"1": "Fixed."}, self.script, 0, 3)
        self.assertEqual((fixes, flags, doubts), ({1: "Fixed."}, {}, {}))
        self.assertEqual(LLMManager._editor_answer([], self.script, 0, 3), ({}, {}, {}))

    def test_editor_line_shows_hearings(self):
        line = {"text_en": "It's Robux!", "speaker": "A", "asr_unsure": True, "asr_hearings": {"prompted": "Robo-X"}}
        out = LLMManager._editor_line(0, line, {"A": {"name": "Chris"}})
        self.assertTrue(out.startswith("L_0 (Chris"))
        self.assertIn('|| UNSURE; prompted: "Robo-X"', out)

    def test_apply_flags(self):
        apply_editor_flags(self.script, {"0": "guessed", "2": "hallucination"}, {"1": "odd"}, {"1": "x"})
        self.assertEqual([s["asr"] for s in self.script], ["guessed", "fixed", "hallucination"])
        self.assertEqual(self.script[1]["speaker_doubt"], "odd")
        self.script[0]["asr_unsure"] = True
        apply_editor_flags(self.script, {}, {})
        self.assertEqual(self.script[0]["asr"], "unsure")

    def test_hallucinations_are_dropped_and_reindexed(self):
        script = [
            {"index": i, "start": i, "text_en": t, "asr": a}
            for i, (t, a) in enumerate([("a", "sure"), ("b", "hallucination"), ("c", "fixed")])
        ]
        kept = DubbingPipeline._drop_hallucinations(script)
        self.assertEqual([(s["index"], s["text_en"]) for s in kept], [(0, "a"), (1, "c")])


class TestPipelineHelpers(unittest.TestCase):
    def test_ref_file_names_carry_the_class(self):
        self.assertEqual(ref_file_name("SPEAKER_03", "calm", 1), "ref_SPEAKER_03.wav")
        self.assertEqual(ref_file_name("SPEAKER_03", "shout", 2), "ref_SPEAKER_03_shout_2.wav")
        for cls in ("calm", "lively", "shout"):
            for n in (1, 2):
                self.assertEqual(ref_file_energy("/x/" + ref_file_name("SPEAKER_03", cls, n)), cls)
        self.assertIsNone(ref_file_energy("/x/designed_SPEAKER_03.wav"))

    def test_script_names(self):
        script = [{"text_en": "Chuck Pitzer drives Sewer Snake."}, {"text_en": "Chuck Pitzer wins!"}]
        names = script_names(script, "1\n00:00:01,000 --> 00:00:02,000\nHere is Robo-X and Jenny Taft.\n")
        self.assertEqual(names[0], "Chuck Pitzer")
        self.assertIn("Sewer Snake", names)
        self.assertIn("Jenny Taft", names)

    def test_tiny_slot_takes_the_silence_around(self):
        lines = [
            {"start": 0.0, "end": 1.0, "text": "Hello there friend."},
            {"start": 5.0, "end": 5.12, "text": "Absolutely incredible!"},
            {"start": 9.0, "end": 10.0, "text": "Yes."},
        ]
        DubbingPipeline._fit_tiny_slots(lines)
        natural = count_syllables("Absolutely incredible!", "en") / 5.0
        self.assertAlmostEqual(lines[1]["end"] - lines[1]["start"], natural, delta=0.01)
        self.assertGreaterEqual(lines[1]["start"], lines[0]["end"])
        self.assertLessEqual(lines[1]["end"], lines[2]["start"])
        self.assertEqual((lines[0]["start"], lines[0]["end"]), (0.0, 1.0))

    def test_energy_groups(self):
        scored = [({"energy": "shout"}, 1.0), ({"energy": None}, 0.9), ({"energy": "calm"}, 0.8)]
        with patch("core.pipeline.EMOTION_REFS", True):
            groups = DubbingPipeline._energy_groups(scored)
        self.assertEqual([(c, len(g)) for c, g in groups], [("calm", 2), ("shout", 1)])
        with patch("core.pipeline.EMOTION_REFS", False):
            self.assertEqual(DubbingPipeline._energy_groups(scored), [("calm", scored)])


class TestRefOrderByEnergy(unittest.TestCase):
    def setUp(self):
        self.manager = TTSManager(inference_lock=None, temp_dir="/tmp/tts_test", speaker_refs={}, abort_event=None)
        self.manager.set_speaker_refs(
            {"A": ["calm1.wav", "calm2.wav", "shout1.wav"]},
            {"A": "male"},
            {"calm1.wav": "calm", "calm2.wav": "calm", "shout1.wav": "shout"},
        )
        self.manager.apply_audition({"A": {"calm1.wav": 0.95, "calm2.wav": 0.9, "shout1.wav": 0.85}}, 0.8)
        self.exists = patch("os.path.exists", return_value=True)
        self.exists.start()

    def tearDown(self):
        self.exists.stop()

    def test_shouted_line_starts_from_the_shouted_clip(self):
        with patch("core.tts_manager.EMOTION_REFS", True):
            order = [p for p, _ in self.manager._ref_order("A", "shout")]
            self.assertEqual(order, ["shout1.wav", "LAST_GOOD", "calm1.wav", "calm2.wav"])
            order = [p for p, _ in self.manager._ref_order("A", "calm")]
            self.assertEqual(order, ["calm1.wav", "calm2.wav", "LAST_GOOD", "shout1.wav"])

    def test_without_energy_rank_order(self):
        with patch("core.tts_manager.EMOTION_REFS", True):
            order = [p for p, _ in self.manager._ref_order("A")]
        self.assertEqual(order, ["calm1.wav", "calm2.wav", "shout1.wav", "LAST_GOOD"])


class TestBankClasses(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.work = tempfile.mkdtemp()
        self.mem = SeriesMemory("battlebots", root=self.root)

    def clip(self, name):
        path = os.path.join(self.work, name)
        with open(path, "wb") as f:
            f.write(b"RIFF....WAVE")
        return path

    def test_shouted_clips_keep_their_slots(self):
        clips = [(self.clip(f"calm{i}.wav"), 0.9 - i * 0.01, None, "calm") for i in range(4)]
        clips.append((self.clip("shout.wav"), 0.6, None, "shout"))
        with patch("core.series_memory.EMOTION_REFS", True), patch("core.series_memory.VOICE_BANK_CLASS_CLIPS", 2):
            self.mem.save_voices({"S": {"person": None, "print": [1.0, 0.0], "profile": {}, "clips": clips}})
        energies = sorted(self.mem.clip_energies("V001").values())
        self.assertEqual(energies, ["calm", "calm", "shout"])

    def test_shout_print_finds_the_person(self):
        self.mem.save_voices({"S": {"person": None, "print": [1.0, 0.0, 0.0], "shout_print": [0.0, 1.0, 0.0], "profile": {}}})
        self.mem.save_voices({"T": {"person": None, "print": [0.0, 0.0, 1.0], "profile": {}}})
        self.assertEqual(self.mem.match({"X": [0.05, 1.0, 0.0]}), {"X": "V001"})

    def tearDown(self):
        for d in (self.root, self.work):
            for base, _, files in os.walk(d):
                for f in files:
                    os.remove(os.path.join(base, f))


if __name__ == "__main__":
    unittest.main()
