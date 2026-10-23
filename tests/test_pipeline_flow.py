"""Flow tests for DubbingPipeline: the whole process_video in MOCK_MODE on a generated clip, plus the
step cache, the Stage 5 checkpoint and the final per-language assembly.

The engines run in MOCK_MODE (canned transcription/translation, 1 s of silence from TTS); ffmpeg is real.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from config import TEMP_DIR
from core.diarization_engine import DiarizationManager
from core.llm_engine import LLMManager
from core.pipeline import DubbingPipeline
from core.synchronizer import SegmentSynchronizer
from core.tts_manager import TTSManager
from core.whisper_engine import WhisperManager
from infrastructure.database import Database
from infrastructure.ffmpeg import FFmpegWrapper


def make_clip(path, seconds=6):
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=d={seconds}:s=160x90",
            "-f", "lavfi", "-i", f"sine=f=220:d={seconds}",
            "-shortest", "-c:v", "mpeg4", "-c:a", "aac", "-metadata:s:a:0", "language=eng",
            path, "-y",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def audio_streams(path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", path], capture_output=True, text=True, check=True
    )
    return [s for s in json.loads(res.stdout)["streams"] if s["codec_type"] == "audio"]


def build_pipeline(langs, workdir, db=None):
    abort = threading.Event()
    llm = LLMManager(model_path="unused.gguf", inference_lock=None, target_langs=langs, abort_event=abort)
    tts = TTSManager(inference_lock=None, temp_dir=TEMP_DIR, speaker_refs={}, abort_event=abort)
    return DubbingPipeline(
        llm_manager=llm,
        tts_manager=tts,
        diar_manager=DiarizationManager(inference_lock=None),
        whisper_manager=WhisperManager(inference_lock=None),
        target_langs=langs,
        db=db,
        output_folder=workdir,
        temp_dir=TEMP_DIR,
    )


class TestProcessVideoMock(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        os.makedirs(TEMP_DIR, exist_ok=True)
        self.clip = os.path.join(self.workdir, "clip.mkv")
        make_clip(self.clip)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_adds_one_track_per_language_and_keeps_original(self):
        build_pipeline(["pl", "de"], self.workdir).process_video(self.clip)

        audio = audio_streams(self.clip)
        tracks = {s["tags"].get("title"): s["tags"].get("language") for s in audio}
        self.assertEqual(len(audio), 3)
        self.assertEqual(tracks.get("AI - Polish"), "pl")
        self.assertEqual(tracks.get("AI - German"), "de")
        self.assertIn("eng", tracks.values())
        self.assertTrue(os.path.exists(self.clip + ".bak"))
        self.assertAlmostEqual(FFmpegWrapper.get_duration(self.clip), 6.0, delta=0.5)

    def test_skips_languages_already_present(self):
        pipeline = build_pipeline(["en"], self.workdir)
        with patch("core.pipeline.prep_audio") as prep:
            pipeline.process_video(self.clip)
        prep.assert_not_called()
        self.assertEqual(len(audio_streams(self.clip)), 1)

    def test_cached_global_analysis_keeps_asr_corrections(self):
        db = Database(os.path.join(self.workdir, "queue.db"))
        cached = {"context": {"glossary": {}}, "speakers": {}, "script_fixes": {"0": "Corrected from cache"}}
        db.save_step_result(1, "Stage 3: Global Analysis", "DONE", result_data=cached)
        pipeline = build_pipeline(["pl"], self.workdir, db=db)
        pipeline.llm_manager.analyze_script = MagicMock()
        pipeline.llm_manager.generate_drafts = MagicMock(wraps=pipeline.llm_manager.generate_drafts)

        pipeline.process_video(self.clip, task_id=1)

        pipeline.llm_manager.analyze_script.assert_not_called()
        script = pipeline.llm_manager.generate_drafts.call_args.args[0]
        self.assertEqual(script[0]["text_en"], "Corrected from cache")


class TestRunStep(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.pipeline = build_pipeline(["pl"], tempfile.gettempdir(), db=self.db)

    def test_runs_and_caches_without_previous_result(self):
        self.db.get_step_result.return_value = None
        func = MagicMock(return_value={"x": 1})
        self.assertEqual(self.pipeline._run_step("Stage X", func, 7, "arg"), {"x": 1})
        func.assert_called_once_with("arg")
        self.db.save_step_result.assert_called_once_with(7, "Stage X", "DONE", result_data={"x": 1})

    def test_reuses_cached_result_when_files_exist(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.db.get_step_result.return_value = [f.name, "vocals"]
            func = MagicMock()
            self.assertEqual(self.pipeline._run_step("Stage X", func, 7), [f.name, "vocals"])
            func.assert_not_called()

    def test_reruns_when_cached_file_is_missing(self):
        self.db.get_step_result.return_value = ["/nonexistent/orig.wav"]
        func = MagicMock(return_value=["/new/orig.wav"])
        self.assertEqual(self.pipeline._run_step("Stage X", func, 7), ["/new/orig.wav"])
        func.assert_called_once()

    def test_no_cache_without_task_id(self):
        func = MagicMock(return_value=1)
        self.pipeline._run_step("Stage X", func, None)
        self.db.get_step_result.assert_not_called()
        self.db.save_step_result.assert_not_called()


class TestProductionPool(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.pipeline = build_pipeline(["pl", "de"], self.workdir)
        self.script = [
            {"index": i, "start": i * 3.0, "end": i * 3.0 + 2.0, "speaker": "A", "text_en": f"Line {i}"} for i in range(3)
        ]

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_resumes_from_checkpoint_and_records_progress(self):
        tasks = [(dict(s), lang) for lang in ("pl", "de") for s in self.script]
        results = {"pl": [{"index": 0, "audio_path": "a.wav"}], "de": []}
        checkpoint = os.path.join(self.workdir, "cp.json")
        self.pipeline.synchronizer = MagicMock()
        self.pipeline.synchronizer.process_segment.side_effect = lambda seg, lang, *a: {
            "audio_path": f"{lang}_{seg['index']}.wav",
            "duration": 1.0,
        }

        self.pipeline._run_concurrent_sync(tasks, results, "vocals.wav", self.script, checkpoint)

        done = [(c.args[1], c.args[0]["index"]) for c in self.pipeline.synchronizer.process_segment.call_args_list]
        self.assertNotIn(("pl", 0), done)
        self.assertEqual(len(done), 5)
        self.assertEqual(sorted(r["index"] for r in results["de"]), [0, 1, 2])
        self.assertEqual(results["de"][0]["speaker"], "A")
        with open(checkpoint) as f:
            self.assertEqual(len(json.load(f)["pl"]), 3)

    def test_fails_instead_of_muxing_a_language_without_any_segment(self):
        self.pipeline.temp_dir = self.workdir
        self.pipeline.llm_manager.generate_drafts = MagicMock(side_effect=lambda script, *a: [dict(s) for s in script])
        self.pipeline.synchronizer = MagicMock()
        self.pipeline.synchronizer.process_segment.side_effect = RuntimeError("XTTS Server died during startup.")

        with patch.object(self.pipeline, "_finalize_all_languages") as finalize:
            with self.assertRaisesRegex(RuntimeError, "pl, de"):
                self.pipeline._execute_production_pool(self.script, "vocals.wav", self.workdir, "orig.wav", None)
        finalize.assert_not_called()

    def test_languages_do_not_share_segment_files(self):
        sync = SegmentSynchronizer(MagicMock(), MagicMock(), self.workdir)
        segment = {"index": 0, "text_en": "Hello", "start": 0, "end": 5, "speaker": "A"}
        paths = {}
        for lang in ("pl", "de"):
            raw = os.path.join(self.workdir, f"tts_{lang}.wav")
            with open(raw, "w") as f:
                f.write(lang)
            sync.tts.synthesize_sync.return_value = {"audio_path": raw, "duration": 1.0}
            with patch("core.synchronizer.FFmpegWrapper.get_duration", return_value=1.0):
                paths[lang] = sync.process_segment(dict(segment), lang, "vocals.wav", [], {})["audio_path"]

        self.assertNotEqual(paths["pl"], paths["de"])
        with open(paths["pl"]) as f:
            self.assertEqual(f.read(), "pl")


class TestFinalizeLanguages(unittest.TestCase):
    def setUp(self):
        self.pipeline = build_pipeline(["pl"], tempfile.gettempdir())

    def _finalize(self, items):
        with (
            patch("core.pipeline.audio_processor.trim_and_pad_silence"),
            patch.object(self.pipeline, "_apply_mastering_and_speed") as master,
            patch("core.pipeline.mix_audio") as mix,
        ):
            tracks = self.pipeline._finalize_all_languages({"pl": items}, "orig.wav", 0.0)
        return tracks, master, mix.call_args.args[1]

    def test_overlapping_line_is_pushed_after_previous_one(self):
        items = [
            {"index": 1, "start": 2.0, "end": 4.0, "duration": 1.5, "audio_path": "b.wav", "speaker": "A"},
            {"index": 0, "start": 0.0, "end": 2.0, "duration": 2.5, "audio_path": "a.wav", "speaker": "A"},
        ]
        tracks, _, clips = self._finalize(items)

        self.assertEqual(tracks, [(os.path.join(self.pipeline.temp_dir, "final_pl.ac3"), "pl", "Polish")])
        first_end = clips[0][1] + clips[0][2]
        self.assertAlmostEqual(clips[1][1], first_end)

    def test_speedup_uses_trailing_silence_and_is_capped(self):
        items = [
            {"index": 0, "start": 0.0, "end": 2.0, "max_end": 2.5, "duration": 2.6, "audio_path": "a.wav", "speaker": "A"},
            {"index": 1, "start": 5.0, "end": 7.0, "duration": 4.0, "audio_path": "b.wav", "speaker": "A"},
        ]
        _, master, _ = self._finalize(items)
        speeds = [c.args[3] for c in master.call_args_list]
        self.assertEqual(speeds[0], 1.0)  # 2.6 s fits into 2.5 s of slot + silence within the 0.2 s tolerance
        self.assertEqual(speeds[1], 1.20)


if __name__ == "__main__":
    unittest.main()
