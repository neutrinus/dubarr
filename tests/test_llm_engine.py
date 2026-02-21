import unittest
from unittest.mock import MagicMock, patch
from core.llm_engine import LLMManager
import json


class TestLLMManager(unittest.TestCase):
    def setUp(self):
        # Patch MOCK_MODE to False to allow custom mocking of .llm
        self.mock_mode_patcher = patch("core.llm_engine.MOCK_MODE", False)
        self.mock_mode_patcher.start()

        self.manager = LLMManager(model_path="fake_model.gguf", inference_lock=None, debug_mode=False, target_langs=["pl"])
        self.manager.llm = MagicMock()

    def tearDown(self):
        self.mock_mode_patcher.stop()

    def test_update_stats(self):
        res = {"usage": {"completion_tokens": 100}}
        self.manager._update_stats(res, 2.5)
        self.assertEqual(self.manager.llm_stats["tokens"], 100)
        self.assertEqual(self.manager.llm_stats["time"], 2.5)

    def test_refine_translation_by_duration_negligible(self):
        # Case where diff is negligible, should return current text without calling LLM
        current = "To jest test"
        res = self.manager.refine_translation_by_duration(
            original_text="This is a test", current_text=current, actual_dur=5.1, target_dur=5.0, glossary={}
        )
        self.assertEqual(res, current)
        self.manager.llm.assert_not_called()

    def test_refine_translation_by_duration_call(self):
        self.manager.llm.return_value = {"choices": [{"text": json.dumps({"final_text": "Krótki test"})}]}
        res = self.manager.refine_translation_by_duration(
            original_text="This is a very long test",
            current_text="To jest bardzo długi test który nie pasuje",
            actual_dur=10.0,
            target_dur=5.0,
            glossary={},
        )
        self.assertEqual(res, "Krótki test")
        self.manager.llm.assert_called_once()

    def test_generate_drafts(self):
        script = [
            {"index": 0, "speaker": "A", "text_en": "Hello", "start": 0, "end": 2},
            {"index": 1, "speaker": "B", "text_en": "(Laughter)", "start": 2, "end": 3},  # Should be skipped
        ]

        self.manager.llm.return_value = {"choices": [{"text": json.dumps({"translations": [{"id": 0, "text": "Cześć"}]})}]}

        drafts = self.manager.generate_drafts(script, "pl", {}, {})

        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["text"], "Cześć")
        self.assertEqual(drafts[0]["index"], 0)

    def test_analyze_script(self):
        script = [{"speaker": "A", "text_en": "Hello world"}]
        # Mock analysis response
        analysis_resp = json.dumps(
            {"summary": "Greeting", "glossary": {"world": "świat"}, "speakers": {"A": {"name": "Alice", "desc": "Female"}}}
        )
        # Mock correction response
        correction_resp = json.dumps({"L_0": "Hello, world!"})

        self.manager.llm.side_effect = [{"choices": [{"text": analysis_resp}]}, {"choices": [{"text": correction_resp}]}]

        with patch("os.makedirs"):
            res = self.manager.analyze_script(script, "debug_dir")

        self.assertEqual(res["context"]["summary"], "Greeting")
        self.assertEqual(res["speakers"]["A"]["name"], "Alice")
        self.assertEqual(script[0]["text_en"], "Hello, world!")
