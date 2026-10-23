import os
import subprocess
import time
import requests
import logging
import signal
import threading

from config import MOCK_MODE

# XTTS v2 supported languages
XTTS_LANGS = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
]


class TTSServerClient:
    """Runs a TTS model server in its own virtual environment and talks to it over HTTP."""

    NAME = "TTS"
    PYTHON_PATH = ""
    SERVER_SCRIPT = ""
    DEFAULT_PORT = 5050

    def __init__(self, gpu_id=0, port=None):
        self.gpu_id = gpu_id
        port = port or self.DEFAULT_PORT
        self.port = port
        self.server_url = f"http://127.0.0.1:{port}"
        self.server_process = None
        self._log_thread = None
        self._restart_lock = threading.Lock()

    def _capture_logs(self):
        """Thread function to capture and log subprocess output."""
        if not self.server_process:
            return

        # Read from stdout and stderr in a way that doesn't block forever
        # and allows the thread to terminate.
        try:
            for line in iter(self.server_process.stdout.readline, ""):
                if line:
                    logging.info(f"[{self.NAME}-Server] {line.strip()}")
                if not self.server_process or self.server_process.poll() is not None:
                    break
        except Exception:
            pass

        try:
            for line in iter(self.server_process.stderr.readline, ""):
                if line:
                    logging.error(f"[{self.NAME}-Server-Err] {line.strip()}")
                if not self.server_process or self.server_process.poll() is not None:
                    break
        except Exception:
            pass

    def start_server(self):
        """Starts the model server in its virtual environment if not already running."""
        with self._restart_lock:
            if MOCK_MODE:
                logging.info(f"{self.NAME}: MOCK_MODE enabled. Skipping server start.")
                return True

            # Check if already running on this port
            try:
                res = requests.get(f"{self.server_url}/health", timeout=2)
                if res.status_code == 200:
                    logging.info(f"{self.NAME}: Server is already running on {self.server_url}. Skipping start.")
                    return True
            except Exception:
                pass

            logging.info(f"{self.NAME}: Starting TTS server...")

            env = os.environ.copy()
            env["GPU_TTS"] = str(self.gpu_id)
            env["TTS_PORT"] = str(self.port)
            # Force immediate output flushing for better log capture
            env["PYTHONUNBUFFERED"] = "1"

            python_path = self.PYTHON_PATH
            server_path = self.SERVER_SCRIPT

            # Capture both stdout and stderr
            self.server_process = subprocess.Popen(
                [python_path, server_path],
                env=env,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Start log capture thread
            self._log_thread = threading.Thread(target=self._capture_logs, daemon=True)
            self._log_thread.start()

            # Wait for server to be ready
            max_retries = 300
            for i in range(max_retries):
                # Check if process is still alive
                if self.server_process and self.server_process.poll() is not None:
                    # Process died unexpectedly
                    try:
                        stderr_output = self.server_process.stderr.read()
                        logging.error(f"{self.NAME}: Server process died early. Stderr: {stderr_output}")
                    except Exception:
                        pass
                    raise RuntimeError(f"{self.NAME} Server died during startup.")

                try:
                    res = requests.get(f"{self.server_url}/health", timeout=1)
                    if res.status_code == 200:
                        logging.info(f"{self.NAME}: Server is ready.")
                        return True
                except Exception:
                    pass
                if i % 10 == 0:
                    logging.info(f"{self.NAME}: Waiting for server... ({i}/{max_retries})")
                time.sleep(1)

            raise RuntimeError(f"{self.NAME} Server failed to start.")

    def sanitize_text(self, text: str) -> str:
        """Removes or replaces characters known to crash XTTS v2."""
        if not text:
            return "..."
        # Replace ellipsis and other special chars
        replacements = {
            "…": "...",
            "—": "-",
            "–": "-",
            "„": '"',
            "”": '"',
            "«": '"',
            "»": '"',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Ensure it's not just whitespace
        if not text.strip():
            return "..."

        return text.strip()

    def _build_payload(self, text, ref_audio, output_path, language, max_duration):
        return {
            "text": text,
            "ref_audio": ref_audio,
            "output_path": output_path,
            "language": language,
        }

    def synthesize(self, text, ref_audio, output_path, language="en", max_duration=None, retry_on_cuda=True):
        """Sends a synthesis request to the model server."""
        text = self.sanitize_text(text)
        payload = self._build_payload(text, ref_audio, output_path, language, max_duration)

        try:
            logging.info(f"{self.NAME}Client: Sending request for text: {text[:20]}...")
            res = requests.post(f"{self.server_url}/synthesize", json=payload, timeout=60)
            logging.info(f"{self.NAME}Client: Received response {res.status_code}")
            if res.status_code != 200:
                error_text = res.text
                logging.error(f"{self.NAME} Server returned {res.status_code}: {error_text}")

                # Detect Tensor Size Mismatch (XTTS v2 specific internal error) - This is input related, do NOT restart
                if "size of tensor" in error_text and "must match" in error_text:
                    raise ValueError(f"XTTS Tensor Error: {error_text}")

                # For ALL other 500 errors (CUDA assert, sens error, unknown internal bugs), RESTART.
                if retry_on_cuda:
                    logging.warning(f"{self.NAME}: Server failure ({res.status_code}). Requesting synchronized restart...")
                    self._handle_restart()
                    logging.info(f"{self.NAME}: Server restarted. Retrying synthesis...")
                    return self.synthesize(
                        text, ref_audio, output_path, language=language, max_duration=max_duration, retry_on_cuda=False
                    )

                raise RuntimeError(f"{self.NAME} Server Error: {error_text}")
        except Exception as e:
            # Also catch connection errors which might occur if the server process crashed completely
            if retry_on_cuda and isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                logging.warning(f"{self.NAME}: Connection error ({e}). Requesting synchronized restart...")
                self._handle_restart()
                return self.synthesize(
                    text, ref_audio, output_path, language=language, max_duration=max_duration, retry_on_cuda=False
                )

            logging.error(f"{self.NAME} Synthesis Failed: {e}")
            raise e

    def _handle_restart(self):
        """Synchronized restart of the server."""
        with self._restart_lock:
            # Check if another thread already restarted it
            try:
                res = requests.get(f"{self.server_url}/health", timeout=1)
                if res.status_code == 200:
                    logging.info(f"{self.NAME}: Server is already back up (restarted by another thread).")
                    return
            except Exception:
                pass

            self.stop_server()
            time.sleep(2)
            self.start_server()

    def stop_server(self):
        """Stops the legacy TTS server only if we started it in this instance."""
        # Note: We don't use the lock here as it might be called from start_server or handle_restart which already has it.
        if self.server_process:
            try:
                logging.info(f"{self.NAME}: Stopping legacy TTS server (PID {self.server_process.pid})...")
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
            except Exception:
                pass
            self.server_process = None
            self._log_thread = None


class XTTSClient(TTSServerClient):
    """XTTS v2 (coqui-tts) server, see src/tts_server.py."""

    NAME = "XTTS"
    PYTHON_PATH = "/app/.venv_tts/bin/python3"
    SERVER_SCRIPT = "tts_server.py"
    DEFAULT_PORT = 5050

    def _build_payload(self, text, ref_audio, output_path, language, max_duration):
        # XTTS cannot target a duration; the synchronizer shortens the text instead
        xtts_lang = language if language in XTTS_LANGS else "en"
        return super()._build_payload(text, ref_audio, output_path, xtts_lang, max_duration)


class OmniVoiceClient(TTSServerClient):
    """OmniVoice (k2-fsa) server, see src/omnivoice_server.py."""

    NAME = "OmniVoice"
    PYTHON_PATH = "/app/.venv_omnivoice/bin/python3"
    SERVER_SCRIPT = "omnivoice_server.py"
    DEFAULT_PORT = 5051

    def _build_payload(self, text, ref_audio, output_path, language, max_duration):
        payload = super()._build_payload(text, ref_audio, output_path, language, max_duration)
        payload["max_duration"] = max_duration
        payload["ref_text"] = read_ref_text(ref_audio)
        return payload


def ref_text_path(ref_audio: str) -> str:
    """Sidecar file holding the transcript of a reference clip."""
    return os.path.splitext(ref_audio)[0] + ".txt"


def write_ref_text(ref_audio: str, text: str):
    if text and text.strip():
        with open(ref_text_path(ref_audio), "w", encoding="utf-8") as f:
            f.write(text.strip())


def read_ref_text(ref_audio: str):
    """Returns the transcript of a reference clip, or None if there is none."""
    try:
        with open(ref_text_path(ref_audio), encoding="utf-8") as f:
            return f.read().strip() or None
    except (OSError, TypeError):
        return None


TTS_CLIENTS = {
    "xtts": XTTSClient,
    "omnivoice": OmniVoiceClient,
}


class TTSWrapper:
    """Starts the server of the chosen TTS backend and forwards synthesis requests."""

    def __init__(self, gpu_id=0, backend="xtts"):
        if backend not in TTS_CLIENTS:
            raise ValueError(f"Unknown TTS backend '{backend}'. Choose one of: {', '.join(TTS_CLIENTS)}")
        self.client = TTS_CLIENTS[backend](gpu_id=gpu_id)
        self.client.start_server()

    def synthesize(self, text, ref_audio, output_path, language="en", max_duration=None):
        self.client.synthesize(text, ref_audio, output_path, language=language, max_duration=max_duration)

    def __del__(self):
        if hasattr(self, "client"):
            self.client.stop_server()


class XTTSWrapper(TTSWrapper):
    """Compatibility wrapper for XTTS v2 Backend."""

    def __init__(self, gpu_id=0):
        super().__init__(gpu_id=gpu_id, backend="xtts")
