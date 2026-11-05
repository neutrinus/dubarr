import os
import subprocess
import time
import requests
import logging
import signal
import threading

from config import MOCK_MODE, OMNIVOICE_PYTHON, TTS_OOM_WAIT, TTS_REF_TEXT, TTS_TIMEOUT


class TTSOutOfMemory(RuntimeError):
    """The TTS server found no free GPU memory, even after waiting TTS_OOM_WAIT seconds."""


def is_out_of_memory(text: str) -> bool:
    return "out of memory" in (text or "").lower()


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

    @staticmethod
    def _stderr_level(line: str) -> int:
        """Python logging and werkzeug write everything to stderr; only real problems are errors."""
        head = line.lstrip()[:12].upper()
        if head.startswith(("INFO", "DEBUG")):
            return logging.INFO
        if head.startswith("WARN") or "WARNING" in line[:60]:
            return logging.WARNING
        return logging.ERROR

    def _pump(self, stream, label: str, is_stderr: bool):
        """Logs one output stream of the server until it closes."""
        try:
            for line in iter(stream.readline, ""):
                line = line.strip()
                if line:
                    level = self._stderr_level(line) if is_stderr else logging.INFO
                    logging.log(level, f"[{self.NAME}-{label}] {line}")
        except Exception:
            pass

    def _capture_logs(self):
        """Starts one reader thread per stream.

        Reading them one after the other left stderr unread while the server ran: every request logs a line there, and
        a full pipe (64 KB, a few hundred requests) blocks the server on its next write.
        """
        if not self.server_process:
            return
        for stream, label, is_stderr in (
            (self.server_process.stdout, "Server", False),
            (self.server_process.stderr, "Server-Err", True),
        ):
            if stream is not None:
                threading.Thread(target=self._pump, args=(stream, label, is_stderr), daemon=True).start()

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
            if not os.path.isfile(python_path):
                raise RuntimeError(
                    f"{self.NAME}: Python interpreter '{python_path}' not found. Check OMNIVOICE_PYTHON "
                    "(by default the TTS server runs on the app's own interpreter)."
                )
            server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.SERVER_SCRIPT)

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
        """Normalizes typographic punctuation and never sends empty text to the model."""
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

    def _build_payload(self, text, ref_audio, output_path, language, max_duration, target_duration=None):
        return {
            "text": text,
            "ref_audio": ref_audio,
            "output_path": output_path,
            "language": language,
        }

    def _post(self, path, payload):
        """POST to the server; while it answers "CUDA out of memory", waits and asks again.

        Memory runs out when another process (a shared llama-server) grows on the same card. Restarting the server does
        not free that memory, and a failed take would count against the line's voice reference; so the request waits
        (5 s, 10 s, 20 s, then 30 s steps) for up to TTS_OOM_WAIT seconds and then raises TTSOutOfMemory.
        """
        waited, pause = 0.0, 5.0
        while True:
            res = requests.post(f"{self.server_url}{path}", json=payload, timeout=TTS_TIMEOUT)
            if res.status_code == 200 or not is_out_of_memory(res.text):
                return res
            if waited >= TTS_OOM_WAIT:
                raise TTSOutOfMemory(f"{self.NAME}: GPU out of memory for {waited:.0f}s: {res.text[:200]}")
            logging.warning(f"{self.NAME}: GPU out of memory, waiting {pause:.0f}s for another process to free it")
            time.sleep(pause)
            waited += pause
            pause = min(pause * 2, 30.0)

    def synthesize(
        self, text, ref_audio, output_path, language="en", max_duration=None, retry_on_cuda=True, target_duration=None
    ):
        """Sends a synthesis request to the model server.

        ``target_duration`` is how long the original speaks, ``max_duration`` the latest the take may end.
        """
        text = self.sanitize_text(text)
        payload = self._build_payload(text, ref_audio, output_path, language, max_duration, target_duration)

        try:
            logging.info(f"{self.NAME}Client: Sending request for text: {text[:20]}...")
            res = self._post("/synthesize", payload)
            logging.info(f"{self.NAME}Client: Received response {res.status_code}")
            if res.status_code != 200:
                error_text = res.text
                logging.error(f"{self.NAME} Server returned {res.status_code}: {error_text}")

                # For ALL 500 errors (CUDA assert, sens error, unknown internal bugs), RESTART.
                if retry_on_cuda:
                    logging.warning(f"{self.NAME}: Server failure ({res.status_code}). Requesting synchronized restart...")
                    self._handle_restart()
                    logging.info(f"{self.NAME}: Server restarted. Retrying synthesis...")
                    return self.synthesize(
                        text,
                        ref_audio,
                        output_path,
                        language=language,
                        max_duration=max_duration,
                        retry_on_cuda=False,
                        target_duration=target_duration,
                    )

                raise RuntimeError(f"{self.NAME} Server Error: {error_text}")
        except TTSOutOfMemory:
            raise
        except Exception as e:
            # Also catch connection errors which might occur if the server process crashed completely
            if retry_on_cuda and isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                logging.warning(f"{self.NAME}: Connection error ({e}). Requesting synchronized restart...")
                self._handle_restart()
                return self.synthesize(
                    text,
                    ref_audio,
                    output_path,
                    language=language,
                    max_duration=max_duration,
                    retry_on_cuda=False,
                    target_duration=target_duration,
                )

            logging.error(f"{self.NAME} Synthesis Failed: {e}")
            raise e

    def design(self, text, instruct, output_path, language="en"):
        """Synthesizes the text in a voice described by attributes (OmniVoice voice design); True on success."""
        payload = {"text": self.sanitize_text(text), "instruct": instruct, "output_path": output_path, "language": language}
        try:
            res = self._post("/design", payload)
            if res.status_code == 200:
                return True
            logging.warning(f"{self.NAME}: design returned {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logging.warning(f"{self.NAME}: design failed: {e}")
        return False

    def estimate(self, texts, ref_audio):
        """Estimated natural duration (seconds) of each text with this voice, or None if the server cannot tell."""
        payload = {
            "texts": [self.sanitize_text(t) for t in texts],
            "ref_audio": ref_audio,
            "ref_text": read_ref_text(ref_audio) if TTS_REF_TEXT == "auto" else "",
        }
        try:
            res = self._post("/estimate", payload)
            if res.status_code == 200:
                return res.json().get("durations")
            logging.warning(f"{self.NAME}: estimate returned {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logging.warning(f"{self.NAME}: estimate failed: {e}")
        return None

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
        """Stops the TTS server only if we started it in this instance."""
        # Note: We don't use the lock here as it might be called from start_server or handle_restart which already has it.
        if self.server_process:
            try:
                logging.info(f"{self.NAME}: Stopping TTS server (PID {self.server_process.pid})...")
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
            except Exception:
                pass
            self.server_process = None
            self._log_thread = None


class OmniVoiceClient(TTSServerClient):
    """OmniVoice (k2-fsa) server, see src/omnivoice_server.py."""

    NAME = "OmniVoice"
    PYTHON_PATH = OMNIVOICE_PYTHON
    SERVER_SCRIPT = "omnivoice_server.py"
    DEFAULT_PORT = 5051

    def _build_payload(self, text, ref_audio, output_path, language, max_duration, target_duration=None):
        payload = super()._build_payload(text, ref_audio, output_path, language, max_duration, target_duration)
        payload["max_duration"] = max_duration
        payload["target_duration"] = target_duration
        # Cross-lingual cloning: the reference clip is in the source language, and sending its transcript makes
        # OmniVoice apply that language's pronunciation to the target text ("walka" heard as "łalka").
        # An empty string means "no transcript" and keeps the target language's phonetics.
        payload["ref_text"] = read_ref_text(ref_audio) if TTS_REF_TEXT == "auto" else ""
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


class TTSWrapper:
    """Starts the OmniVoice server and forwards synthesis requests."""

    def __init__(self, gpu_id=0):
        self.client = OmniVoiceClient(gpu_id=gpu_id)
        self.client.start_server()

    def synthesize(self, text, ref_audio, output_path, language="en", max_duration=None, target_duration=None):
        self.client.synthesize(
            text, ref_audio, output_path, language=language, max_duration=max_duration, target_duration=target_duration
        )

    def estimate(self, texts, ref_audio):
        return self.client.estimate(texts, ref_audio)

    def design(self, text, instruct, output_path, language="en"):
        return self.client.design(text, instruct, output_path, language=language)

    def __del__(self):
        if hasattr(self, "client"):
            self.client.stop_server()
