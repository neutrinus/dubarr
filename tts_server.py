import os
import logging
from flask import Flask, request, jsonify
from TTS.api import TTS
import torch

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

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')
    ref_audio = data.get('ref_audio')
    output_path = data.get('output_path')
    language = data.get('language', 'en')
    
    try:
        # Standard XTTS synthesis
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_audio,
            language=language,
            file_path=output_path
        )
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"XTTS Synthesis Failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    load_tts()
    # Run on a local port
    app.run(host='127.0.0.1', port=5050)
