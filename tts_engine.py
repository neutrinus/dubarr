import os
import subprocess
import time
import requests
import logging
import signal

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

    def start_server(self):
        """Starts the XTTS server in the legacy virtual environment."""
        logging.info("XTTS: Starting legacy TTS server...")

        env = os.environ.copy()
        env["GPU_TTS"] = str(self.gpu_id)

        # Use the absolute path to the legacy venv python
        python_path = "/app/.venv_tts/bin/python3"
        server_path = "/app/tts_server.py"

        self.server_process = subprocess.Popen(
            [python_path, server_path], env=env, preexec_fn=os.setsid
        )

        # Wait for server to be ready
        max_retries = 120
        for i in range(max_retries):
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

    def synthesize(self, text, ref_audio, output_path, language="en"):
        """Sends a synthesis request to the legacy server."""
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
            res = requests.post(f"{self.server_url}/synthesize", json=payload, timeout=120)
            if res.status_code != 200:
                raise RuntimeError(f"XTTS Server Error: {res.text}")
        except Exception as e:
            logging.error(f"XTTS Synthesis Failed: {e}")
            raise e

    def stop_server(self):
        """Stops the legacy TTS server."""
        if self.server_process:
            try:
                logging.info("XTTS: Stopping legacy TTS server...")
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            except Exception:
                pass
            self.server_process = None


class F5TTSWrapper:
    """Compatibility wrapper for XTTS v2 Backend."""

    def __init__(self, gpu_id=0):
        self.client = XTTSClient(gpu_id=gpu_id)
        self.client.start_server()

    def synthesize(self, text, ref_audio, output_path, ref_text="", language="en"):
        self.client.synthesize(text, ref_audio, output_path, language=language)

    def __del__(self):
        if hasattr(self, "client"):
            self.client.stop_server()
