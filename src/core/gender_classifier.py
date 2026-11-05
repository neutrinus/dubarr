"""Gender of a voice by a trained classifier (Spec 002, 2.3): the last resort when nothing else tells.

The model (audeering/wav2vec2-large-robust-24-ft-age-gender, CC-BY-NC-SA 4.0) reads the whole timbre, not only the
pitch, which made men women in 9 of 11 verdicts. It runs on the CPU (JUDGE_DEVICE), is loaded on first use and
released after the gender check.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from config import GENDER_MODEL, HF_TOKEN, JUDGE_DEVICE, MOCK_MODE

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with the dev environment
    np = None

logger = logging.getLogger(__name__)

RATE = 16000
LABELS = ("female", "male", "child")  # the model's output order
MAX_CLIP_SECONDS = 10.0


def _build_model_class():
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            if hasattr(self, "post_init"):
                self.post_init()
            else:  # pragma: no cover - older transformers
                self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

    return AgeGenderModel


class GenderClassifier:
    def __init__(self, model_name: str = GENDER_MODEL, device: str = JUDGE_DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = None  # False when it could not be loaded
        self._processor = None

    @property
    def enabled(self) -> bool:
        return bool(self.model_name) and self.model_name != "off" and not MOCK_MODE

    def _load(self) -> bool:
        if self._model is None:
            try:
                from transformers import Wav2Vec2Processor

                self._processor = Wav2Vec2Processor.from_pretrained(self.model_name, token=HF_TOKEN)
                model = _build_model_class().from_pretrained(self.model_name, token=HF_TOKEN).eval()
                self._model = model.to(self.device) if self.device != "cpu" else model
                logger.info(f"Gender classifier: '{self.model_name}' loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Gender classifier: '{self.model_name}' did not load ({e}); no voice-based gender")
                self._model = False
        return bool(self._model)

    def probabilities(self, samples) -> Optional[Dict[str, float]]:
        """{female, male, child} probabilities of one 16 kHz mono clip; None when it cannot tell."""
        if not self.enabled or np is None or samples is None or len(samples) < RATE or not self._load():
            return None
        import torch

        clip = np.asarray(samples[: int(MAX_CLIP_SECONDS * RATE)], dtype=np.float32)
        try:
            values = self._processor(clip, sampling_rate=RATE).input_values[0]
            with torch.no_grad():
                _, gender = self._model(torch.tensor(np.asarray(values)[None, :]).to(self._model.device))
            probs = gender[0].float().cpu().numpy()
        except Exception as e:
            logger.warning(f"Gender classifier failed: {e}")
            return None
        return {label: float(p) for label, p in zip(LABELS, probs)}

    def classify(self, clips: Sequence) -> Tuple[Optional[str], float]:
        """("male" | "female" | None, confidence) of a speaker from several clips: the mean of their probabilities.

        A child's voice counts as neither: its gender is not told by the voice.
        """
        found: List[Dict[str, float]] = [p for p in (self.probabilities(c) for c in clips) if p]
        if not found:
            return None, 0.0
        mean = {label: sum(p[label] for p in found) / len(found) for label in LABELS}
        best = max(mean, key=mean.get)
        if best == "child":
            return None, mean[best]
        return best, mean[best]

    def shutdown(self):
        self._model = None
        self._processor = None
        import gc

        gc.collect()


def speaker_clips(script: List[Dict], samples, spk: str, max_clips: int = 6) -> List:
    """Up to ``max_clips`` of the speaker's calm lines from the (unfiltered) vocals, longest first, 1.5-10 s each.

    Calm, because shouting moves any voice toward the other gender's range (S01E03).
    """
    if samples is None:
        return []
    lines = [s for s in script if s.get("speaker") == spk and 1.5 <= s["end"] - s["start"]]
    calm = [s for s in lines if (s.get("energy") or "calm") == "calm"] or lines
    calm.sort(key=lambda s: s["end"] - s["start"], reverse=True)
    out = []
    for s in calm[:max_clips]:
        a, b = int(s["start"] * RATE), int(min(s["end"], s["start"] + MAX_CLIP_SECONDS) * RATE)
        if b - a >= RATE:
            out.append(samples[a:b])
    return out
