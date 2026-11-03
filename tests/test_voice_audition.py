import unittest
from unittest.mock import MagicMock, patch

from core.synchronizer import SegmentSynchronizer
from core.tts_manager import TTSManager
from core.voice_audition import audition_voices, pick_test_line


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
        self.assertEqual(pick_test_line(drafts, "A", "pl"), "Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia.")

    def test_no_line_for_a_speaker_with_only_interjections(self):
        self.assertIsNone(pick_test_line([{"speaker": "A", "text": "O!"}], "A", "pl"))


class TestAuditionVoices(unittest.TestCase):
    def setUp(self):
        self.tts = TTSManager(inference_lock=None, temp_dir="/tmp", speaker_refs={}, abort_event=MagicMock())
        self.tts.set_speaker_refs({"ANN": ["shout.wav", "calm.wav"]}, {"ANN": "male"})
        self.tts._run_synthesis = MagicMock()
        self.drafts = [{"speaker": "ANN", "text": "Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia."}]

    def run_audition(self, heard, durations):
        checker = MagicMock(side_effect=heard)
        with (
            patch("core.voice_audition.os.path.exists", return_value=True),
            patch("core.voice_audition.os.remove"),
            patch("core.voice_audition.FFmpegWrapper.get_duration", side_effect=durations),
        ):
            return audition_voices(self.tts, checker, self.drafts, "pl", "/tmp", SegmentSynchronizer.is_swallowed), checker

    def test_ranks_the_candidate_that_clones_intelligibly_first(self):
        scores, _ = self.run_audition(["Naf! Naf!", "Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia."], [3.0, 3.2])

        self.assertLess(scores["ANN"]["shout.wav"], 0.5)
        self.assertGreater(scores["ANN"]["calm.wav"], 0.9)
        self.tts.apply_audition(scores, 0.8)
        self.assertEqual(self.tts.speaker_refs["ANN"], ["calm.wav", "shout.wav"])
        self.assertNotIn("ANN", self.tts.line_refs_off)

    def test_swallowed_audition_take_scores_zero_without_asking_whisper(self):
        scores, checker = self.run_audition(["Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia."], [0.4, 3.2])

        self.assertEqual(scores["ANN"]["shout.wav"], 0.0)
        self.assertEqual(checker.call_count, 1)

    def test_synthesis_error_scores_zero(self):
        self.tts._run_synthesis.side_effect = [RuntimeError("HTTP 500"), None]
        scores, _ = self.run_audition(["Ajs-wejw zadaje Rejzor-bakowi poważne obrażenia."], [3.2])

        self.assertEqual(scores["ANN"], {"shout.wav": 0.0, "calm.wav": 1.0})

    def test_speaker_is_left_out_when_the_checker_cannot_tell(self):
        scores, _ = self.run_audition([None, None], [3.0, 3.2])

        self.assertEqual(scores, {})


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
        self.assertIsNone(voice_instruct({"gender": "unknown", "desc": "Crowd"}))

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
