import os
import glob
import subprocess
import logging
import torch
import gc
import shutil
import time
from typing import List, Dict, Tuple
from config import GPU_AUDIO, TEMP_DIR, WHISPER_MODEL
from utils import run_cmd

def prep_audio(vpath: str) -> Tuple[str, str]:
    a_stereo = os.path.join(TEMP_DIR, "orig.wav")
    run_cmd(["ffmpeg", "-i", vpath, "-vn", "-ac", "2", "-y", a_stereo], "extract audio")
    run_cmd(["demucs", "--mp3", "--two-stems", "vocals", "-o", TEMP_DIR, "-n",
                  "mdx_extra_q", "--device", f"cuda:{GPU_AUDIO}", a_stereo], "demucs separation")
    found = glob.glob(os.path.join(TEMP_DIR, "**", "vocals.mp3"), recursive=True)
    if not found:
        raise FileNotFoundError("Demucs failed to produce vocals.mp3")
    return a_stereo, found[0]

def analyze_audio(vocals_path: str, gpu_index: int) -> Tuple[List, List, Dict]:
    mpath = os.path.join(TEMP_DIR, "mono.wav")
    run_cmd(["ffmpeg", "-i", vocals_path, "-ac", "1", mpath, "-y"], "mono conversion")
    
    diar_result = []
    trans_result = []
    durations = {}
    
    def run_diar():
        t0 = time.perf_counter()
        from pyannote.audio import Pipeline
        logging.info(f"Diarization: Loading on GPU {gpu_index}")
        p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ.get("HF_TOKEN"))
        p.to(torch.device(f"cuda:{gpu_index}"))
        res = p(mpath)
        for s, _, label in res.itertracks(yield_label=True):
            diar_result.append({"start": s.start, "end": s.end, "speaker": label})
        durations["2a. Diarization (Parallel)"] = time.perf_counter() - t0
        del p
        gc.collect()
        torch.cuda.empty_cache()

    def run_whisper():
        t0 = time.perf_counter()
        from faster_whisper import WhisperModel
        logging.info(f"Whisper: Loading on GPU {gpu_index}")
        m = WhisperModel(WHISPER_MODEL, device="cuda", device_index=gpu_index, compute_type="float16")
        ts, _ = m.transcribe(mpath)
        for x in ts:
            if x.avg_logprob < -1.0:
                continue
            trans_result.append({
                "start": x.start, 
                "end": x.end, 
                "text": x.text.strip(), 
                "avg_logprob": x.avg_logprob, 
                "no_speech_prob": x.no_speech_prob, 
                "compression_ratio": x.compression_ratio
            })
        durations["2b. Transcription (Sequential)"] = time.perf_counter() - t0
        del m
        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"Starting Sequential Audio Analysis on GPU {gpu_index}...")
    run_diar()
    run_whisper()
    
    return diar_result, trans_result, durations

def trim_silence(path: str):
    tmp = path + ".trim.wav"
    filter_chain = "areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse"
    cmd = ["ffmpeg", "-i", path, "-af", filter_chain, tmp, "-y"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.getsize(tmp) > 100:
            os.replace(tmp, path)
    except Exception:
        pass
    if os.path.exists(tmp):
        os.remove(tmp)

def mix_audio(bg: str, clips: List, out: str):
    if not clips: 
        shutil.copy(bg, out)
        return
    filter_path = os.path.join(TEMP_DIR, "mix_filter.txt")
    inputs = ["-i", bg]
    filter_str = ""
    for i, (path, start, duration) in enumerate(clips):
        inputs.extend(["-i", path])
        delay_ms = int(start * 1000)
        fade_st = max(0, duration - 0.01)
        filter_str += f"[{i+1}:a]afade=t=in:st=0:d=0.01,afade=t=out:st={fade_st:.3f}:d=0.01,adelay={delay_ms}|{delay_ms}[a{i+1}];"
    
    filter_str += "".join([f"[a{i+1}]" for i in range(len(clips))]) + f"amix=inputs={len(clips)}:normalize=0[speech_raw];"
    filter_str += "[speech_raw]asplit=2[speech_out][trigger];[0:a][trigger]sidechaincompress=threshold=0.02:ratio=5:attack=50:release=600[bg_ducked];[bg_ducked][speech_out]amix=inputs=2:weights=1 1.2:normalize=0[out]"
    
    with open(filter_path, "w") as f: 
        f.write(filter_str)
    
    run_cmd(["ffmpeg"] + inputs + ["-filter_complex_script", filter_path, "-map", "[out]", "-c:a", "ac3", out, "-y"], "final mixing")

def mux_video(v: str, a: str, lang: str, out: str, lang_name: str):
    title = f"AI - {lang_name}"
    subprocess.run([
        "ffmpeg", "-i", v, "-i", a, 
        "-map", "0:v", "-map", "1:a", "-map", "0:a", 
        "-c:v", "copy", "-c:a", "ac3", 
        "-metadata:s:a:0", f"language={lang}", 
        "-metadata:s:a:0", f"title={title}", 
        out, "-y"
    ], capture_output=True)

def extract_clean_segment(input_path: str, start: float, end: float, output_path: str):
    """Extracts and cleans a specific audio segment for Voice Cloning."""
    duration = end - start
    # Filter: Highpass to remove rumble, afftdn for noise reduction, speechnorm for consistent volume
    filt = "highpass=f=100,afftdn=nf=-20,speechnorm=e=10:r=0.0001:l=1"
    cmd = [
        "ffmpeg", "-v", "error", 
        "-ss", str(start), 
        "-t", str(duration), 
        "-i", input_path, 
        "-af", filt, 
        "-ac", "1", 
        "-ar", "22050", 
        output_path, "-y"
    ]
    subprocess.run(cmd, check=False, capture_output=True)
