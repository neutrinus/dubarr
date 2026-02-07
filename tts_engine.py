import os
import torch
import logging
import soundfile as sf
import numpy as np
import sys

# Try importing F5-TTS modules
try:
    import f5_tts
    from f5_tts.model import DiT, CFM
    from f5_tts.infer.utils_infer import (
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
        infer_process,
    )
    HAS_F5 = True
except ImportError as e:
    HAS_F5 = False
    logging.error(f"F5-TTS import failed: {e}")
    # Try adding local path if installed from git clone but not in path
    if "/app/F5-TTS/src" not in sys.path:
        sys.path.append("/app/F5-TTS/src")
        try:
            from f5_tts.model import DiT, CFM
            from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
            HAS_F5 = True
            logging.info("F5-TTS imported from local source /app/F5-TTS/src")
        except ImportError as e2:
            logging.error(f"F5-TTS local import also failed: {e2}")

class F5TTSWrapper:
    def __init__(self, gpu_id=0, model_name="F5-TTS"):
        if not HAS_F5:
            raise RuntimeError("F5-TTS library is missing.")
        
        self.device = f"cuda:{gpu_id}"
        logging.info(f"F5-TTS: Initializing on {self.device}...")
        
        # Load Vocoder (Vocos)
        self.vocoder = load_vocoder(is_local=False, device=self.device)
        
        # Download model from HF
        from huggingface_hub import hf_hub_download
        logging.info("F5-TTS: Downloading/Checking model from Hugging Face...")
        ckpt_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/model_1200000.safetensors")
        
        # Load Model using high-level utility
        logging.info(f"F5-TTS: Loading model from {ckpt_path}...")
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_path,
            vocab_file="/app/F5-TTS/src/f5_tts/infer/examples/vocab.txt",
            device=self.device
        )
        self.model.eval()
        logging.info("F5-TTS: Model Loaded.")

    def synthesize(self, text, ref_audio, output_path, ref_text=""):
        """Synthesize speech using F5-TTS."""
        if not text: return
        
        try:
            # Main Inference
            audio, sr, spect = infer_process(
                ref_audio, 
                ref_text, 
                text, 
                self.model, 
                self.vocoder, 
                device=self.device, 
                speed=1.0
            )
            
            # Save output
            sf.write(output_path, audio, sr)
            
        except Exception as e:
            logging.error(f"F5-TTS Synthesis Failed: {e}")
            raise e
