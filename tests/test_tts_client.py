import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from infrastructure.tts_client import OmniVoiceClient, TTSWrapper, XTTSClient, XTTSWrapper, write_ref_text
import requests


class TestTTSClient(unittest.TestCase):
    def setUp(self):
        self.mock_mode_patcher = patch("infrastructure.tts_client.MOCK_MODE", False)
        self.mock_mode_patcher.start()
        self.client = XTTSClient(gpu_id=0, port=5050)

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


class TestXTTSWrapper(unittest.TestCase):
    @patch("infrastructure.tts_client.XTTSClient.start_server")
    def test_wrapper_init(self, mock_start):
        wrapper = XTTSWrapper(gpu_id=1)
        mock_start.assert_called_once()
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
    def test_payload_has_duration_and_ref_text(self, mock_post):
        mock_post.return_value.status_code = 200
        ref = os.path.join(self.tmp, "ref_A.wav")
        write_ref_text(ref, "  Hello there.  ")
        self.client.synthesize("Cześć", ref, "out.wav", "pl", max_duration=2.5)
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(self.client.port, 5051)
        self.assertEqual(payload["max_duration"], 2.5)
        self.assertEqual(payload["ref_text"], "Hello there.")
        self.assertEqual(payload["language"], "pl")

    @patch("requests.post")
    def test_payload_without_transcript(self, mock_post):
        mock_post.return_value.status_code = 200
        self.client.synthesize("Cześć", os.path.join(self.tmp, "none.wav"), "out.wav", "pl")
        self.assertIsNone(mock_post.call_args[1]["json"]["ref_text"])

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            TTSWrapper(backend="nope")

    @patch("requests.post")
    def test_xtts_ignores_duration(self, mock_post):
        mock_post.return_value.status_code = 200
        XTTSClient().synthesize("text", "ref.wav", "out.wav", "sv", max_duration=2.0)
        payload = mock_post.call_args[1]["json"]
        self.assertNotIn("max_duration", payload)
        self.assertEqual(payload["language"], "en")
