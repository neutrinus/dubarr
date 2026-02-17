from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import os


def verify_audio(path):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    # Check signal
    data, sr = sf.read(path)
    print(f"File: {path}")
    print(f"Shape: {data.shape}, SR: {sr}, Dtype: {data.dtype}")
    print(f"Max: {np.max(data)}, Min: {np.min(data)}, Mean: {np.mean(data)}")

    if np.max(np.abs(data)) < 0.001:
        print("WARNING: Audio is nearly silent!")
        return

    # Run Whisper
    print("Running Whisper on segment...")
    model = WhisperModel("base", device="cuda", compute_type="float16")
    segments, info = model.transcribe(path)
    print(f"Detected language: {info.language} ({info.language_probability:.2f})")

    text = " ".join([s.text for s in segments])
    print(f"Transcription: '{text}'")


if __name__ == "__main__":
    # Test one segment from the last run
    verify_audio("output/debug_myth_test/segments/en_tts_0.wav")
