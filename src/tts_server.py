import os
import sys
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    output_path = data.get("output_path")
    # Mock synthesis: create a 1s silent file
    import subprocess
    cmd = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1", "-y", output_path]
    subprocess.run(cmd, capture_output=True, check=True)
    return jsonify({"status": "done"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="127.0.0.1", port=port)
