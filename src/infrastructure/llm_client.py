import logging

import requests


class OpenAICompatibleClient:
    """Chat completions against an OpenAI-compatible server (e.g. llama.cpp's llama-server)."""

    def __init__(self, base_url: str, model: str = "", api_key: str = "", timeout: int = 600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def chat(self, messages, **kwargs) -> dict:
        """Returns the raw response: {"choices": [{"message": {"content": ...}}], "usage": {...}}."""
        payload = {"messages": messages, **kwargs}
        if self.model:
            payload["model"] = self.model
        resp = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
        if resp.status_code != 200:
            logging.error(f"LLM server returned {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
        return resp.json()
