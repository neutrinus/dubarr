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

    def test_generate_drafts_passes_show_context_and_speaker_gender(self):
        script = [{"index": 0, "speaker": "S0", "text_en": "I did it", "start": 0, "end": 2}]
        speakers = {"S0": {"name": "Una", "gender": "female", "desc": "Commander"}}
        context = {"media": 'Star Trek (2022), S03E03 "Shuttle"', "summary": "A crash.", "glossary": {}}
        self.chat.return_value = reply(json.dumps({"translations": [{"id": 0, "text": "Zrobiłam to"}]}))

        drafts = self.manager.generate_drafts(script, "pl", speakers, context)

        messages = self.chat.call_args.kwargs["messages"]
        self.assertIn('Star Trek (2022), S03E03 "Shuttle"\nA crash.', messages[0]["content"])
        batch = json.loads(messages[1]["content"])["translate"]
        self.assertEqual(batch[0]["speaker"], "Una")
        self.assertEqual(batch[0]["speaker_gender"], "female")
        self.assertEqual(drafts[0]["speaker_label"], "Una (female)")

    def test_speaker_gender_falls_back_to_profile(self):
        from core.llm_engine import speaker_gender

        self.assertEqual(speaker_gender({"A": {"desc": "Middle-aged man, deep voice"}}, "A"), "male")
        self.assertEqual(speaker_gender({"A": {"gender": "unknown", "desc": "Female, 30s"}}, "A"), "female")
        self.assertEqual(speaker_gender({"A": {"desc": "Narrator"}}, "A"), "")
        self.assertEqual(speaker_gender({}, "A"), "")

    def test_analyze_script_includes_media_info(self):
        self.chat.side_effect = [reply(json.dumps({"summary": "s", "glossary": {}, "speakers": {}})), reply("{}")]
        self.manager.analyze_script([{"speaker": "A", "text_en": "Hi"}], "d", "", "TV series: Star Trek")
        self.assertIn("KNOWN MEDIA INFO", self.chat.call_args_list[0].kwargs["messages"][0]["content"])
        self.assertIn("TV series: Star Trek", self.chat.call_args_list[0].kwargs["messages"][0]["content"])

    def test_analyze_script_gets_names_and_fixed_glossary(self):
        self.chat.side_effect = [reply(json.dumps({"summary": "s", "glossary": {}, "speakers": {}})), reply("{}")]
        self.manager.analyze_script(
            [{"speaker": "A", "text_en": "Hi"}], "d", "", "", {"Razorback": "Rejzorbek"}, ["Razorback", "Ice Wave"]
        )
        system = self.chat.call_args_list[0].kwargs["messages"][0]["content"]
        self.assertIn('"Razorback": "Rejzorbek"', system)
        self.assertIn("Razorback, Ice Wave", system)
        self.assertIn("Razorback, Ice Wave", self.chat.call_args_list[1].kwargs["messages"][0]["content"])

    def test_glossary_reading_test_rejects_a_lost_consonant(self):
        # "Rej-or-bek" sounds like "Rayorbeck": the "z" of Razorback is gone, so it is respelled and read again
        self.chat.side_effect = [
            reply(json.dumps({"heard": [{"id": 0, "english": "Rayorbeck"}, {"id": 1, "english": "Hineman"}]})),
            reply(json.dumps({"spellings": [{"id": 0, "spelling": "Rejzorbek"}]})),
            reply(json.dumps({"heard": [{"id": 0, "english": "Rayzorbeck"}]})),
        ]
        res = self.manager.check_glossary({"Razorback": "Rej-or-bek", "Hyneman": "Hajneman"})
        self.assertEqual(res, {"Razorback": "Rejzorbek", "Hyneman": "Hajneman"})
        respell = json.loads(self.chat.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(respell["names"][0]["sounds_like"], "Rayorbeck")

    def test_glossary_keeps_the_original_when_every_spelling_fails(self):
        bad = reply(json.dumps({"heard": [{"id": 0, "english": "Rayorbeck"}]}))
        respell = reply(json.dumps({"spellings": [{"id": 0, "spelling": "Rejorbak"}]}))
        self.chat.side_effect = [bad, respell, bad, respell, bad]
        self.assertEqual(self.manager.check_glossary({"Razorback": "Rej-or-bek"}), {"Razorback": "Razorback"})

    def test_glossary_fixed_spellings_win_and_skip_the_test(self):
        res = self.manager.check_glossary({"Razorback": "Rej-or-bek", "Ice Wave": "Ice Wave"}, {"Razorback": "Rejzorbek"})
        self.assertEqual(res, {"Razorback": "Rejzorbek", "Ice Wave": "Ice Wave"})
        self.chat.assert_not_called()

    def test_analyze_script(self):
        script = [{"speaker": "A", "text_en": "Hello world"}]
        # Mock analysis response
        analysis_resp = json.dumps(
            {"summary": "Greeting", "glossary": {"World": "World"}, "speakers": {"A": {"name": "Alice", "desc": "Female"}}}
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

    def test_retries_truncated_answer_with_double_budget(self):
        truncated = reply("")
        truncated["choices"][0]["finish_reason"] = "length"
        complete = reply(json.dumps({"final_text": "Krótko"}))
        complete["choices"][0]["finish_reason"] = "stop"
        self.chat.side_effect = [truncated, complete]

        res = self.manager.refine_translation_by_duration("Long line", "Długa linia", 10.0, 5.0, {})

        self.assertEqual(res, "Krótko")
        self.assertEqual(self.chat.call_count, 2)
        first, second = self.chat.call_args_list
        self.assertEqual(second.kwargs["max_tokens"], first.kwargs["max_tokens"] * 2)

    def test_drafts_use_the_configured_budget(self):
        self.chat.return_value = reply(json.dumps({"translations": [{"id": 0, "text": "Cześć"}]}))
        self.manager.generate_drafts([{"speaker": "A", "text_en": "Hello", "start": 0, "end": 2}], "pl", {}, {})

        self.assertEqual(self.chat.call_args.kwargs["max_tokens"], self.manager_max_tokens())

    @staticmethod
    def manager_max_tokens():
        from core.llm_engine import LLM_MAX_TOKENS

        return LLM_MAX_TOKENS

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
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(self.manager.llm_stats["tokens"], 5)


class TestWindowedTranslation(unittest.TestCase):
    def setUp(self):
        patcher = patch("core.llm_engine.MOCK_MODE", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = LLMManager(model_path="fake_model.gguf", inference_lock=None, debug_mode=False, target_langs=["pl"])
        self.manager.llm = MagicMock()
        self.chat = self.manager.llm.create_chat_completion

    def test_lines_get_a_syllable_budget_from_speech_and_pause(self):
        budget = LLMManager.syllable_budget({"start": 0.0, "end": 2.0, "max_end": 3.0})
        self.assertEqual(budget["target_syllables"], 10)
        self.assertEqual(budget["max_syllables"], 15)
        self.assertLess(budget["min_syllables"], 10)

    def test_next_window_sees_previous_translations(self):
        script = [{"speaker": "A", "text_en": f"Line {i}", "start": i, "end": i + 1} for i in range(25)]

        def answer(messages, **kwargs):
            request = json.loads(messages[1]["content"])
            return reply(
                json.dumps({"translations": [{"id": x["id"], "text": f"Linia {x['id']}"} for x in request["translate"]]})
            )

        self.chat.side_effect = answer
        with patch("core.llm_engine.TRANSLATION_WINDOW", 20):
            drafts = self.manager.generate_drafts(script, "pl", {}, {})

        self.assertEqual([d["text"] for d in drafts][:2], ["Linia 0", "Linia 1"])
        second = json.loads(self.chat.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual([x["id"] for x in second["translate"]], list(range(20, 25)))
        self.assertEqual(second["previous"][-1], {"speaker": "A", "text": "Line 19", "translation": "Linia 19"})
        first = json.loads(self.chat.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertEqual(first["next"][0]["text"], "Line 20")

    def test_review_replaces_only_fixed_lines_within_budget(self):
        drafts = [
            {"index": 0, "speaker": "A", "text_en": "They have two mini-bots.", "text": "Mają.", "start": 0, "end": 3},
            {"index": 1, "speaker": "A", "text_en": "Hi.", "text": "Cześć.", "start": 3, "end": 3.5},
        ]
        self.chat.return_value = reply(
            json.dumps(
                {
                    "problems": ["L0 lost the mini-bots"],
                    "fixes": [
                        {"id": 0, "text": "Mają dwa miniboty."},
                        {"id": 1, "text": "Cześć wszystkim, bardzo serdecznie witamy was tutaj dzisiaj na arenie."},
                    ],
                }
            )
        )

        reviewed = self.manager.review_drafts(drafts, "pl", {}, {})

        self.assertEqual(reviewed[0]["text"], "Mają dwa miniboty.")
        self.assertEqual(reviewed[1]["text"], "Cześć.")  # far over its 0.5 s budget

    def test_drafts_keep_length_variants_and_review_drops_stale_ones(self):
        script = [{"speaker": "A", "text_en": "They have two mini-bots.", "start": 0, "end": 3}]
        self.chat.return_value = reply(
            json.dumps(
                {
                    "translations": [
                        {"id": 0, "text": "Mają dwa miniboty.", "short": "Dwa miniboty.", "long": "Mają aż dwa miniboty."}
                    ]
                }
            )
        )
        drafts = self.manager.generate_drafts(script, "pl", {}, {})
        self.assertEqual(drafts[0]["variants"], ["Mają dwa miniboty.", "Dwa miniboty.", "Mają aż dwa miniboty."])

        self.chat.return_value = reply(json.dumps({"fixes": [{"id": 0, "text": "Mają dwa małe miniboty."}]}))
        reviewed = self.manager.review_drafts(drafts, "pl", {}, {})
        self.assertEqual(reviewed[0]["variants"], ["Mają dwa małe miniboty."])
