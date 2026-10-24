import unittest
from unittest.mock import MagicMock, patch
from core.llm_engine import LLMManager
import json


def reply(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"completion_tokens": 5}}


class TestLLMManager(unittest.TestCase):
    def setUp(self):
        # Patch MOCK_MODE to False to allow custom mocking of .llm
        self.mock_mode_patcher = patch("core.llm_engine.MOCK_MODE", False)
        self.mock_mode_patcher.start()

        self.manager = LLMManager(model_path="fake_model.gguf", inference_lock=None, debug_mode=False, target_langs=["pl"])
        self.manager.llm = MagicMock()
        self.chat = self.manager.llm.create_chat_completion

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
        self.chat.assert_not_called()

    def test_refine_translation_by_duration_call(self):
        self.chat.return_value = reply(json.dumps({"final_text": "Krótki test"}))
        res = self.manager.refine_translation_by_duration(
            original_text="This is a very long test",
            current_text="To jest bardzo długi test który nie pasuje",
            actual_dur=10.0,
            target_dur=5.0,
            glossary={},
        )
        self.assertEqual(res, "Krótki test")
        self.chat.assert_called_once()

    def test_generate_drafts(self):
        script = [
            {"index": 0, "speaker": "A", "text_en": "Hello", "start": 0, "end": 2},
            {"index": 1, "speaker": "B", "text_en": "(Laughter)", "start": 2, "end": 3},  # Should be skipped
        ]

        self.chat.return_value = reply(json.dumps({"translations": [{"id": 0, "text": "Cześć"}]}))

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

        self.chat.side_effect = [reply(analysis_resp), reply(correction_resp)]

        with patch("os.makedirs"):
            res = self.manager.analyze_script(script, "debug_dir")

        self.assertEqual(res["context"]["summary"], "Greeting")
        self.assertEqual(res["speakers"]["A"]["name"], "Alice")
        self.assertEqual(script[0]["text_en"], "Hello, world!")
        self.assertEqual(res["script_fixes"], {"0": "Hello, world!"})

    def test_sends_chat_messages_with_json_output(self):
        self.chat.return_value = reply(json.dumps({"translations": [{"id": 0, "text": "Cześć"}]}))
        self.manager.generate_drafts([{"speaker": "A", "text_en": "Hello", "start": 0, "end": 2}], "pl", {}, {})

        kwargs = self.chat.call_args.kwargs
        self.assertEqual([m["role"] for m in kwargs["messages"]], ["system", "user"])
        self.assertIn("Polish", kwargs["messages"][0]["content"])
        self.assertNotIn("<|im_start|>", kwargs["messages"][0]["content"])
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


class TestExternalServer(unittest.TestCase):
    def setUp(self):
        patches = [
            patch("core.llm_engine.MOCK_MODE", False),
            patch("core.llm_engine.LLM_API_URL", "http://llama:8080/v1"),
            patch("core.llm_engine.LLM_API_MODEL", "gemma-4"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.manager = LLMManager(model_path="unused.gguf", inference_lock=None, target_langs=["pl"])

    def test_uses_server_without_loading_a_model(self):
        with patch("infrastructure.llm_client.requests.post") as post, patch("core.llm_engine.Llama") as llama:
            post.return_value = MagicMock(status_code=200, json=MagicMock(return_value=reply('{"final_text": "Krótko"}')))
            res = self.manager.refine_translation_by_duration("Long line", "Długa linia", 10.0, 5.0, {})

        self.assertEqual(res, "Krótko")
        llama.assert_not_called()
        self.assertEqual(self.manager.status, "READY")
        url = post.call_args.args[0]
        body = post.call_args.kwargs["json"]
        self.assertEqual(url, "http://llama:8080/v1/chat/completions")
        self.assertEqual(body["model"], "gemma-4")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(self.manager.llm_stats["tokens"], 5)
