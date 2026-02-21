import os
import logging
import torch
from flask import Flask, request, jsonify
from TTS.api import TTS

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

tts = None
gpu_id = int(os.environ.get("GPU_TTS", 0))


def load_tts():
    global tts
    logging.info(f"XTTS: Loading model on cuda:{gpu_id}...")
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(f"cuda:{gpu_id}")
    logging.info("XTTS: Model loaded.")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    text = data.get("text", "").strip()
    ref_audio = data.get("ref_audio")
    output_path = data.get("output_path")
    language = data.get("language", "en")

    # Critical fix: XTTS crashes on empty or very short non-alphanumeric text
    if not text or len(text) < 1 or not any(c.isalnum() for c in text):
        logging.warning(f"XTTS: Received empty or invalid text: '{text}'. Returning silent dummy.")
        # Create a silent wav using ffmpeg or just return success with an existing silent file if we had one.
        # For now, let's try to synthesize just a dot which is safe.
        text = "."

    try:
        # Standard XTTS synthesis
        tts.tts_to_file(text=text, speaker_wav=ref_audio, language=language, file_path=output_path)

        # Free GPU memory after each call to prevent fragmentation/crashes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({"status": "success"})
    except Exception as e:
        err_msg = str(e)
        logging.error(f"XTTS Synthesis Failed: {err_msg}")

        # Free GPU memory even on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Specific handling for the 'sens' library bug to avoid 500 if possible,
        # but here we must report it so the client can restart if needed.
        return jsonify({"status": "error", "message": err_msg}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    load_tts()
    # Run on a local port
    app.run(host="127.0.0.1", port=5050)
