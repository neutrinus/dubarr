import os
import logging
from collections import OrderedDict

import soundfile as sf
import torch
from flask import Flask, request, jsonify
from omnivoice import OmniVoice

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

model = None
gpu_id = int(os.environ.get("GPU_TTS", 0))

# OmniVoice pads every clip with this much silence per side (OmniVoiceGenerationConfig.pad_duration)
PAD_DURATION = 0.1
# Never speed speech up by more than this to fit a slot; longer lines go back to the LLM to be shortened
MAX_SPEEDUP = 1.25
PROMPT_CACHE_SIZE = 64
_prompt_cache = OrderedDict()


def load_model():
    global model
    if torch.cuda.is_available():
        device, dtype = f"cuda:{gpu_id}", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    logging.info(f"OmniVoice: Loading model on {device}...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    logging.info("OmniVoice: Model loaded.")


def get_voice_prompt(ref_audio, ref_text):
    """Encodes a reference clip once and reuses it for every line spoken with it."""
    key = (ref_audio, os.path.getmtime(ref_audio), ref_text)
    prompt = _prompt_cache.get(key)
    if prompt is None:
        if not ref_text:
            logging.warning(f"OmniVoice: No transcript for {ref_audio}, the model will transcribe it with Whisper.")
        prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
        _prompt_cache[key] = prompt
        if len(_prompt_cache) > PROMPT_CACHE_SIZE:
            _prompt_cache.popitem(last=False)
    else:
        _prompt_cache.move_to_end(key)
    return prompt


def pick_duration(text, prompt, max_duration):
    """Returns the duration to request, or None to let the model speak at its natural pace.

    Lines that naturally fit are left alone. Lines that run over are squeezed into the slot,
    but by at most MAX_SPEEDUP, so speech never sounds rushed.
    """
    if not max_duration:
        return None
    budget = max(0.3, max_duration - 2 * PAD_DURATION)
    frame_rate = model.audio_tokenizer.config.frame_rate
    natural = model._estimate_target_tokens(text, prompt.ref_text, prompt.ref_audio_tokens.size(-1)) / frame_rate
    if natural <= budget:
        return None
    return max(budget, natural / MAX_SPEEDUP)


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    text = data.get("text", "").strip()
    ref_audio = data.get("ref_audio")
    ref_text = data.get("ref_text")
    output_path = data.get("output_path")
    language = data.get("language")
    max_duration = data.get("max_duration")

    if not text or not any(c.isalnum() for c in text):
        logging.warning(f"OmniVoice: Received empty or invalid text: '{text}'. Synthesizing a dot.")
        text = "."

    try:
        prompt = get_voice_prompt(ref_audio, ref_text)
        duration = pick_duration(text, prompt, max_duration)
        audio = model.generate(text=text, language=language, voice_clone_prompt=prompt, duration=duration)
        sf.write(output_path, audio[0], model.sampling_rate)
        return jsonify({"status": "success", "duration": duration})
    except Exception as e:
        logging.error(f"OmniVoice Synthesis Failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=int(os.environ.get("TTS_PORT", 5051)))
