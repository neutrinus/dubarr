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


class XTTSClient:
    def __init__(self, gpu_id=0, port=5050):
        self.gpu_id = gpu_id
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
                    logging.info(f"[XTTS-Server] {line.strip()}")
                if not self.server_process or self.server_process.poll() is not None:
                    break
        except Exception:
            pass

        try:
            for line in iter(self.server_process.stderr.readline, ""):
                if line:
                    logging.error(f"[XTTS-Server-Err] {line.strip()}")
                if not self.server_process or self.server_process.poll() is not None:
                    break
        except Exception:
            pass

    def start_server(self):
        """Starts the XTTS server in the legacy virtual environment if not already running."""
        with self._restart_lock:
            if MOCK_MODE:
                logging.info("XTTS: MOCK_MODE enabled. Skipping server start.")
                return True

            # Check if already running on this port
            try:
                res = requests.get(f"{self.server_url}/health", timeout=2)
                if res.status_code == 200:
                    logging.info(f"XTTS: Server is already running on {self.server_url}. Skipping start.")
                    return True
            except Exception:
                pass

            logging.info("XTTS: Starting legacy TTS server...")

            env = os.environ.copy()
            env["GPU_TTS"] = str(self.gpu_id)
            # Force immediate output flushing for better log capture
            env["PYTHONUNBUFFERED"] = "1"

            # Use the absolute path to the legacy venv python
            python_path = "/app/.venv_tts/bin/python3"
            server_path = "tts_server.py"

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
                        logging.error(f"XTTS: Server process died early. Stderr: {stderr_output}")
                    except Exception:
                        pass
                    raise RuntimeError("XTTS Server died during startup.")

                try:
                    res = requests.get(f"{self.server_url}/health", timeout=1)
                    if res.status_code == 200:
                        logging.info("XTTS: Server is ready.")
                        return True
                except Exception:
                    pass
                if i % 10 == 0:
                    logging.info(f"XTTS: Waiting for server... ({i}/{max_retries})")
                time.sleep(1)

            raise RuntimeError("XTTS Server failed to start.")

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

    def synthesize(self, text, ref_audio, output_path, language="en", retry_on_cuda=True):
        """Sends a synthesis request to the legacy server."""
        text = self.sanitize_text(text)

        # Map language to XTTS supported
        if language == "pl":
            xtts_lang = "pl"
        elif language == "en":
            xtts_lang = "en"
        elif language == "ko":
            xtts_lang = "ko"
        elif language == "ja":
            xtts_lang = "ja"
        elif language in XTTS_LANGS:
            xtts_lang = language
        else:
            xtts_lang = "en"

        payload = {
            "text": text,
            "ref_audio": ref_audio,
            "output_path": output_path,
            "language": xtts_lang,
        }

        try:
            res = requests.post(f"{self.server_url}/synthesize", json=payload, timeout=60)
            if res.status_code != 200:
                error_text = res.text
                logging.error(f"XTTS Server returned {res.status_code}: {error_text}")

                # Detect Tensor Size Mismatch (XTTS v2 specific internal error) - This is input related, do NOT restart
                if "size of tensor" in error_text and "must match" in error_text:
                    raise ValueError(f"XTTS Tensor Error: {error_text}")

                # For ALL other 500 errors (CUDA assert, sens error, unknown internal bugs), RESTART.
                if retry_on_cuda:
                    logging.warning(f"XTTS: Server failure ({res.status_code}). Requesting synchronized restart...")
                    self._handle_restart()
                    logging.info("XTTS: Server restarted. Retrying synthesis...")
                    return self.synthesize(text, ref_audio, output_path, language=language, retry_on_cuda=False)

                raise RuntimeError(f"XTTS Server Error: {error_text}")
        except Exception as e:
            # Also catch connection errors which might occur if the server process crashed completely
            if retry_on_cuda and isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                logging.warning(f"XTTS: Connection error ({e}). Requesting synchronized restart...")
                self._handle_restart()
                return self.synthesize(text, ref_audio, output_path, language=language, retry_on_cuda=False)

            logging.error(f"XTTS Synthesis Failed: {e}")
            raise e

    def _handle_restart(self):
        """Synchronized restart of the server."""
        with self._restart_lock:
            # Check if another thread already restarted it
            try:
                res = requests.get(f"{self.server_url}/health", timeout=1)
                if res.status_code == 200:
                    logging.info("XTTS: Server is already back up (restarted by another thread).")
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
                logging.info(f"XTTS: Stopping legacy TTS server (PID {self.server_process.pid})...")
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
            except Exception:
                pass
            self.server_process = None
            self._log_thread = None


class XTTSWrapper:
    """Compatibility wrapper for XTTS v2 Backend."""

    def __init__(self, gpu_id=0):
        self.client = XTTSClient(gpu_id=gpu_id)
        self.client.start_server()

    def synthesize(self, text, ref_audio, output_path, language="en"):
        self.client.synthesize(text, ref_audio, output_path, language=language)

    def __del__(self):
        if hasattr(self, "client"):
            self.client.stop_server()
