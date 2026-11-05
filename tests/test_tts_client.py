import logging
import time
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
from infrastructure.tts_client import OmniVoiceClient, TTSOutOfMemory, TTSWrapper, write_ref_text
import requests


class TestTTSClient(unittest.TestCase):
    def setUp(self):
        self.mock_mode_patcher = patch("infrastructure.tts_client.MOCK_MODE", False)
        self.mock_mode_patcher.start()
        self.client = OmniVoiceClient(gpu_id=0, port=5051)
        self.client.PYTHON_PATH = sys.executable

    def tearDown(self):
        self.mock_mode_patcher.stop()

    @patch("requests.get")
    def test_health_check_running(self, mock_get):
        mock_get.return_value.status_code = 200
        # start_server should return True if server already running
        with patch("subprocess.Popen") as mock_popen:
            res = self.client.start_server()
            self.assertTrue(res)
            mock_popen.assert_not_called()

    @patch("requests.get")
    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_start_server_success(self, mock_sleep, mock_popen, mock_get):
        # First get fails (server not running)
        # Second get (after Popen) succeeds
        mock_get.side_effect = [requests.exceptions.ConnectionError(), MagicMock(status_code=200)]

        # Mock Popen return value
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        mock_process.stdout.readline.return_value = ""  # Terminate log thread loop
        mock_process.stderr.readline.return_value = ""  # Terminate log thread loop
        mock_popen.return_value = mock_process

        res = self.client.start_server()
        self.assertTrue(res)
        mock_popen.assert_called_once()
        self.assertEqual(mock_popen.call_args[0][0][0], sys.executable)

    @patch("requests.get", side_effect=requests.exceptions.ConnectionError())
    @patch("subprocess.Popen")
    def test_start_server_missing_interpreter(self, mock_popen, mock_get):
        self.client.PYTHON_PATH = "/nonexistent/bin/python3"
        with self.assertRaisesRegex(RuntimeError, "OMNIVOICE_PYTHON"):
            self.client.start_server()
        mock_popen.assert_not_called()

    def test_omnivoice_runs_on_app_interpreter_by_default(self):
        if "OMNIVOICE_PYTHON" in os.environ:
            self.skipTest("OMNIVOICE_PYTHON is set")
        self.assertEqual(OmniVoiceClient.PYTHON_PATH, sys.executable)

    @patch("requests.get")
    @patch("requests.post")
    def test_synthesize_success(self, mock_post, mock_get):
        mock_post.return_value.status_code = 200
        self.client.synthesize("text", "ref.wav", "out.wav", "pl")
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["language"], "pl")

    @patch("requests.get")
    @patch("requests.post")
    def test_synthesize_error(self, mock_post, mock_get):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Error message"
        # Mock health check to avoid long wait during restart attempt
        mock_get.return_value.status_code = 200
        with self.assertRaises(RuntimeError):
            self.client.synthesize("text", "ref.wav", "out.wav", "pl")


    @patch("infrastructure.tts_client.time.sleep")
    @patch("requests.get")
    @patch("requests.post")
    def test_out_of_memory_waits_instead_of_restarting(self, mock_post, mock_get, mock_sleep):
        # Another process filled the card: the server is fine, so it is not restarted; the request waits and asks again
        oom = MagicMock(status_code=500, text='{"message":"CUDA out of memory. Tried to allocate 20.00 MiB."}')
        ok = MagicMock(status_code=200, text="")
        mock_post.side_effect = [oom, oom, ok]
        with patch.object(self.client, "_handle_restart") as restart:
            self.client.synthesize("text", "ref.wav", "out.wav", "pl")
        restart.assert_not_called()
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual([c.args[0] for c in mock_sleep.call_args_list], [5.0, 10.0])

    @patch("infrastructure.tts_client.TTS_OOM_WAIT", 20)
    @patch("infrastructure.tts_client.time.sleep")
    @patch("requests.post")
    def test_out_of_memory_gives_up_after_the_wait(self, mock_post, mock_sleep):
        mock_post.return_value = MagicMock(status_code=500, text="CUDA out of memory.")
        with patch.object(self.client, "_handle_restart") as restart:
            with self.assertRaises(TTSOutOfMemory):
                self.client.synthesize("text", "ref.wav", "out.wav", "pl")
        restart.assert_not_called()
        self.assertEqual(sum(c.args[0] for c in mock_sleep.call_args_list), 35.0)  # 5 + 10 + 20


class TestTTSWrapper(unittest.TestCase):
    @patch("infrastructure.tts_client.OmniVoiceClient.start_server")
    def test_wrapper_init(self, mock_start):
        wrapper = TTSWrapper(gpu_id=1)
        mock_start.assert_called_once()
        self.assertIsInstance(wrapper.client, OmniVoiceClient)
        self.assertEqual(wrapper.client.gpu_id, 1)


class TestOmniVoiceClient(unittest.TestCase):
    def setUp(self):
        self.mock_mode_patcher = patch("infrastructure.tts_client.MOCK_MODE", False)
        self.mock_mode_patcher.start()
        self.client = OmniVoiceClient(gpu_id=0)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.mock_mode_patcher.stop()
        import shutil

        shutil.rmtree(self.tmp)

    @patch("requests.post")
    def test_payload_has_duration_and_sends_no_transcript_by_default(self, mock_post):
        # The reference clip is in the source language; its transcript would make OmniVoice use that
        # language's pronunciation for the target text ("walka" -> "łalka"), so it is left out.
        mock_post.return_value.status_code = 200
        ref = os.path.join(self.tmp, "ref_A.wav")
        write_ref_text(ref, "  Hello there.  ")
        self.client.synthesize("Cześć", ref, "out.wav", "pl", max_duration=2.5)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(self.client.port, 5051)
        self.assertEqual(payload["max_duration"], 2.5)
        self.assertEqual(payload["ref_text"], "")
        self.assertEqual(payload["language"], "pl")

    @patch("requests.post")
    def test_payload_sends_transcript_when_requested(self, mock_post):
        mock_post.return_value.status_code = 200
        ref = os.path.join(self.tmp, "ref_A.wav")
        write_ref_text(ref, "  Hello there.  ")
        with patch("infrastructure.tts_client.TTS_REF_TEXT", "auto"):
            self.client.synthesize("Cześć", ref, "out.wav", "pl", max_duration=2.5)
        self.assertEqual(mock_post.call_args[1]["json"]["ref_text"], "Hello there.")

    @patch("requests.post")
    def test_payload_without_transcript(self, mock_post):
        mock_post.return_value.status_code = 200
        self.client.synthesize("Cześć", os.path.join(self.tmp, "none.wav"), "out.wav", "pl")
        self.assertEqual(mock_post.call_args[1]["json"]["ref_text"], "")


class TestServerLogs(unittest.TestCase):
    def test_stderr_levels(self):
        from infrastructure.tts_client import TTSServerClient

        level = TTSServerClient._stderr_level
        self.assertEqual(level('INFO:werkzeug:127.0.0.1 - - "POST /synthesize HTTP/1.1" 200 -'), logging.INFO)
        self.assertEqual(level("WARNING:root:slow"), logging.WARNING)
        self.assertEqual(level("Traceback (most recent call last):"), logging.ERROR)

    def test_both_streams_are_read_at_once(self):
        import io
        from infrastructure.tts_client import TTSServerClient

        client = TTSServerClient()
        client.server_process = MagicMock()
        # stdout never ends while the server runs; stderr must be read anyway
        client.server_process.stdout = MagicMock()
        pending = iter(["ready\n"] * 6)
        client.server_process.stdout.readline.side_effect = lambda: time.sleep(0.1) or next(pending, "")
        client.server_process.stderr = io.StringIO("INFO:werkzeug:request\n")
        with self.assertLogs(level="INFO") as logs:
            client._capture_logs()
            time.sleep(0.2)
        self.assertTrue(any("Server-Err] INFO:werkzeug:request" in m for m in logs.output))
