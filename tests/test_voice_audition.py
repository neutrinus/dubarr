import os
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

from core.synchronizer import SegmentSynchronizer
from core.tts_manager import TTSManager
from core.voice_audition import audition_voices, median_pitch, pick_test_line, speaker_pitch


class TestPickTestLine(unittest.TestCase):
    def test_picks_the_speakers_line_closest_to_four_seconds(self):
        drafts = [
            {"speaker": "A", "text": "Wow!"},
            {"speaker": "B", "text": "Zupełnie inna osoba mówi tę kwestię, która ma sporo sylab."},
            {"speaker": "A", "text": "Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia."},
            {
                "speaker": "A",
                "text": "Szczątki latają po całej arenie! To dwudziestokilogramowy element, który pędzi bardzo szybko!",
            },
        ]
        # The line with glossary respellings ("Ajs-wejw") is heard as the English name: a line without one goes first
        self.assertEqual(pick_test_line(drafts, "A", "pl"), drafts[3]["text"])
        self.assertEqual(pick_test_line(drafts[:3], "A", "pl"), drafts[2]["text"])

    def test_no_line_for_a_speaker_with_only_interjections(self):
        self.assertIsNone(pick_test_line([{"speaker": "A", "text": "O!"}], "A", "pl"))


class TestAuditionVoices(unittest.TestCase):
    def setUp(self):
        self.tts = TTSManager(inference_lock=None, temp_dir="/tmp", speaker_refs={}, abort_event=MagicMock())
        self.tts.set_speaker_refs({"ANN": ["shout.wav", "calm.wav"]}, {"ANN": "male"})
        self.tts._run_synthesis = MagicMock()
        self.drafts = [{"speaker": "ANN", "text": "Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia."}]

    def run_audition(self, scores, durations):
        scorer = MagicMock(side_effect=scores)
        with (
            patch("core.voice_audition.os.path.exists", return_value=True),
            patch("core.voice_audition.os.remove"),
            patch("core.voice_audition.FFmpegWrapper.get_duration", side_effect=durations),
        ):
            return audition_voices(self.tts, scorer, self.drafts, "pl", "/tmp", SegmentSynchronizer.is_swallowed), scorer

    def test_ranks_the_candidate_that_clones_best_first(self):
        scores, scorer = self.run_audition([0.2, 0.95], [3.0, 3.2])

        self.assertEqual(scores["ANN"], {"shout.wav": 0.2, "calm.wav": 0.95})
        self.assertEqual(scorer.call_args_list[1].args[1:], (self.drafts[0]["text"], "pl", "calm.wav"))
        self.tts.apply_audition(scores, 0.8)
        self.assertEqual(self.tts.speaker_refs["ANN"], ["calm.wav", "shout.wav"])
        self.assertNotIn("ANN", self.tts.line_refs_off)

    def test_swallowed_audition_take_scores_zero_without_asking_the_judge(self):
        scores, scorer = self.run_audition([0.9], [0.4, 3.2])

        self.assertEqual(scores["ANN"]["shout.wav"], 0.0)
        self.assertEqual(scorer.call_count, 1)

    def test_synthesis_error_scores_zero(self):
        self.tts._run_synthesis.side_effect = [RuntimeError("HTTP 500"), None]
        scores, _ = self.run_audition([1.0], [3.2])

        self.assertEqual(scores["ANN"], {"shout.wav": 0.0, "calm.wav": 1.0})

    def test_full_gpu_leaves_the_candidate_unscored(self):
        # MythBusters: the card stayed full and calm clips scored 0.00, so lively ones jumped ahead of them
        from infrastructure.tts_client import TTSOutOfMemory

        self.tts._run_synthesis.side_effect = [TTSOutOfMemory("CUDA out of memory"), None]
        scores, _ = self.run_audition([1.0], [3.2])

        self.assertEqual(scores["ANN"], {"calm.wav": 1.0})

    def test_speaker_is_left_out_when_the_judge_cannot_tell(self):
        scores, _ = self.run_audition([None, None], [3.0, 3.2])

        self.assertEqual(scores, {})


class TestGenericDesignedVoice(unittest.TestCase):
    def test_speaker_without_profile_or_gender_gets_a_generic_designed_voice(self):
        from core.voice_audition import design_voices

        tts = TTSManager(inference_lock=None, temp_dir="/tmp", speaker_refs={}, abort_event=MagicMock())
        tts.set_speaker_refs({"ANN": ["shout.wav"]})
        tts.engine = MagicMock()
        tts.engine.design.return_value = True
        tts.status = "READY"
        drafts = [
            {"speaker": "ANN", "text": "Gotowi Strażnicy Gór Skalistych, pokażcie swoją moc!"},
            {"speaker": "unknown", "text": "Raz, dwa, trzy, gotowi do startu, zaczynamy!"},
        ]
        with (
            patch("core.voice_audition.os.path.exists", return_value=True),
            patch("core.voice_audition.os.path.getsize", return_value=50000),
            patch("core.voice_audition.write_ref_text"),
        ):
            designed = design_voices(tts, {"ANN": {"gender": "unknown"}}, drafts, "pl", "/tmp")

        self.assertEqual(set(designed), {"ANN", "unknown"})
        # A single line is spoken in the design clip even though it is also the test line
        self.assertEqual([c.args[1] for c in tts.engine.design.call_args_list], ["", ""])


if __name__ == "__main__":
    unittest.main()


class TestVoiceDesign(unittest.TestCase):
    def test_instruct_from_the_profile(self):
        from core.voice_audition import voice_instruct

        self.assertEqual(
            voice_instruct({"gender": "male", "desc": "Middle-aged announcer with a deep, booming voice"}),
            "male, middle-aged, low pitch",
        )
        self.assertEqual(voice_instruct({"gender": "Female", "desc": "Energetic host"}), "female")
        # Nothing known: OmniVoice picks a generic voice
        self.assertEqual(voice_instruct({"gender": "unknown", "desc": "Crowd"}), "")
        # Whole words only: a man telling his childhood story is no child
        self.assertEqual(voice_instruct({"gender": "male", "desc": "Recounts his childhood story"}), "male")
        self.assertEqual(voice_instruct({"gender": "unknown", "desc": "Younger announcer"}), "young adult")

    def test_designed_clip_joins_the_candidates_and_is_never_lent(self):
        from core.voice_audition import design_voices

        tts = TTSManager(inference_lock=None, temp_dir="/tmp", speaker_refs={}, abort_event=MagicMock())
        tts.set_speaker_refs({"ANN": ["shout.wav"], "EXP": ["exp.wav"]}, {"ANN": "male", "EXP": "male", "NEW": "male"})
        tts.engine = MagicMock()
        tts.engine.design.return_value = True
        tts.status = "READY"
        drafts = [
            {"speaker": "ANN", "text": "Czerwony kwadrat prosto z Kolorado, pokażcie im swoją moc!"},
            {"speaker": "ANN", "text": "Czas wysłać rywala prosto do nieba, oto Rejzor-bak!"},
            {"speaker": "NEW", "text": "Nikt tu nie ma czystej próbki mojego głosu, niestety."},
            {"speaker": "NEW", "text": "Ale mam profil i płeć, więc da się mnie opisać."},
        ]
        info = {"ANN": {"gender": "male", "desc": "loud"}, "NEW": {"gender": "male"}, "EXP": {"gender": "male"}}
        with (
            patch("core.voice_audition.os.path.exists", return_value=True),
            patch("core.voice_audition.os.path.getsize", return_value=50000),
            patch("core.voice_audition.write_ref_text"),
        ):
            designed = design_voices(tts, info, drafts, "pl", "/tmp")

        self.assertEqual(set(designed), {"ANN", "NEW"})
        self.assertEqual(tts.speaker_refs["ANN"], ["shout.wav", designed["ANN"]])
        # The designed clip is spoken from a line other than the audition test line
        text = tts.engine.design.call_args_list[0].args[0]
        self.assertNotEqual(text, "")
        with patch("os.path.exists", return_value=True):
            item = {"speaker": "NEW", "index": 1, "start": 0.0, "end": 12.0}
            self.assertEqual(tts._select_reference(item, "v.wav", "pl"), (designed["NEW"], "DESIGNED"))
            item = {"speaker": "EXP", "index": 1, "start": 0.0, "end": 12.0}
            path, _ = tts._select_reference(item, "v.wav", "pl", skip={"exp.wav"})
            self.assertNotIn(path, designed.values())


def voiced(f0, rate=16000, seconds=1.5):
    """A buzzy vowel-like tone: harmonics of f0, as a voice has."""
    np = __import__("numpy")
    t = np.arange(int(rate * seconds)) / rate
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)).astype("float32") * 0.2


class TestPitchGender(unittest.TestCase):
    def setUp(self):
        __import__("pytest").importorskip("numpy")

    def test_median_pitch_of_a_voice(self):
        self.assertAlmostEqual(median_pitch(voiced(120.0), 16000), 120.0, delta=4)
        self.assertAlmostEqual(median_pitch(voiced(220.0), 16000), 220.0, delta=6)
        self.assertIsNone(median_pitch(voiced(120.0)[:400], 16000))  # too short to tell

    def wav(self, samples, rate=16000):
        np = __import__("numpy")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes((samples * 32767).astype(np.int16).tobytes())
        self.addCleanup(os.remove, path)
        return path

    def test_pitch_of_the_reference_clips_is_only_a_number(self):
        self.assertAlmostEqual(speaker_pitch([self.wav(voiced(115.0)), self.wav(voiced(125.0))]), 120.0, delta=6)
        self.assertIsNone(speaker_pitch(["missing.wav"]))
