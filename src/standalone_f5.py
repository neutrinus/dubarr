import sys
import torch
import soundfile as sf
import logging

# Add F5-TTS to path
sys.path.append("/app/F5-TTS/src")

from f5_tts.model import DiT  # noqa: E402
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process  # noqa: E402

logging.basicConfig(level=logging.INFO)


def test_f5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load Vocoder
    vocoder = load_vocoder(is_local=False, device=device)

    # Load Model
    ckpt_path = "/root/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_Base/model_1200000.safetensors"
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=ckpt_path,
        vocab_file="/app/F5-TTS/src/f5_tts/infer/examples/vocab.txt",
        device=device,
    )

    # Reference
    ref_audio = "/app/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "The quick brown fox jumps over the lazy dog."

    logging.info("Starting inference...")
    audio, sr, spect = infer_process(ref_audio, ref_text, gen_text, model, vocoder, device=device)

    logging.info(f"Output Type: {type(audio)}")
    logging.info(f"Output Shape: {audio.shape}")
    logging.info(f"Output Dtype: {audio.dtype}")
    logging.info(f"Sample Rate: {sr}")

    # Save 1: Standard sf.write
    sf.write("/app/output/test_sf.wav", audio, sr)

    # Save 2: Force 1D if not already
    audio_1d = audio.flatten()
    sf.write("/app/output/test_sf_1d.wav", audio_1d, sr)

    # Save 3: Use torchaudio if available
    import torchaudio

    torchaudio.save("/app/output/test_torch.wav", torch.from_numpy(audio).unsqueeze(0), sr)

    logging.info("Files saved to /app/output/")


if __name__ == "__main__":
    test_f5()
