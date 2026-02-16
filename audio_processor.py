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
    """Extracts original stereo audio and separates vocals using Demucs."""
    a_stereo = os.path.join(TEMP_DIR, "orig.wav")
    run_cmd(["ffmpeg", "-i", vpath, "-vn", "-ac", "2", "-y", a_stereo], "extract audio")

    demucs_cmd = [
        "demucs",
        "--mp3",
        "--two-stems",
        "vocals",
        "-o",
        TEMP_DIR,
        "-n",
        "mdx_extra_q",
        "--device",
        f"cuda:{GPU_AUDIO}",
        a_stereo,
    ]
    run_cmd(demucs_cmd, "demucs separation")

    found = glob.glob(os.path.join(TEMP_DIR, "**", "vocals.mp3"), recursive=True)
    if not found:
        raise FileNotFoundError("Demucs failed to produce vocals.mp3")
    return a_stereo, found[0]


def run_diarization(mpath: str, gpu_index: int) -> List[Dict]:
    """Runs speaker diarization using Pyannote."""
    from pyannote.audio import Pipeline

    logging.info(f"Diarization: Loading on GPU {gpu_index}")

    p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ.get("HF_TOKEN"))
    p.to(torch.device(f"cuda:{gpu_index}"))
    res = p(mpath)

    annotation = getattr(res, "speaker_diarization", getattr(res, "diarization", getattr(res, "annotation", res)))

    diar_result = []
    try:
        for s, _, label in annotation.itertracks(yield_label=True):
            diar_result.append({"start": s.start, "end": s.end, "speaker": label})
    except AttributeError:
        logging.error(f"Diarization output {type(res)} has no itertracks.")
        raise

    del p
    gc.collect()
    torch.cuda.empty_cache()
    return diar_result


def run_transcription(mpath: str, gpu_index: int) -> List[Dict]:
    """Runs transcription using Faster-Whisper."""
    from faster_whisper import WhisperModel

    logging.info(f"Whisper: Loading on GPU {gpu_index}")

    m = WhisperModel(WHISPER_MODEL, device="cuda", device_index=gpu_index, compute_type="float16")
    ts, _ = m.transcribe(mpath)

    trans_result = []
    for x in ts:
        if x.avg_logprob < -1.0:
            continue
        trans_result.append(
            {
                "start": x.start,
                "end": x.end,
                "text": x.text.strip(),
                "avg_logprob": x.avg_logprob,
                "no_speech_prob": x.no_speech_prob,
            }
        )

    del m
    gc.collect()
    torch.cuda.empty_cache()
    return trans_result


def analyze_audio(vocals_path: str, gpu_index: int) -> Tuple[List, List, Dict]:
    """Orchestrates diarization and transcription."""
    mpath = os.path.join(TEMP_DIR, "mono.wav")
    run_cmd(["ffmpeg", "-i", vocals_path, "-ac", "1", mpath, "-y"], "mono conversion")

    durations = {}
    logging.info(f"Starting Sequential Audio Analysis on GPU {gpu_index}...")

    t0 = time.perf_counter()
    diar_result = run_diarization(mpath, gpu_index)
    durations["2a. Diarization (Parallel)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    trans_result = run_transcription(mpath, gpu_index)
    durations["2b. Transcription (Sequential)"] = time.perf_counter() - t0

    return diar_result, trans_result, durations


def mix_audio(bg: str, clips: List, out: str):
    """Mixes background audio with dubbed clips using sidechain compression."""
    if not clips:
        shutil.copy(bg, out)
        return

    filter_path = os.path.join(TEMP_DIR, "mix_filter.txt")
    inputs = ["-i", bg]
    filters = []

    for i, (path, start, duration) in enumerate(clips):
        inputs.extend(["-i", path])
        delay_ms = int(start * 1000)
        fade_st = max(0, duration - 0.05)

        # Format each clip: Stereo 48k, fade in/out, delay
        f = (
            f"[{i + 1}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"afade=t=in:st=0:d=0.05,afade=t=out:st={fade_st:.3f}:d=0.05,"
            f"adelay={delay_ms}|{delay_ms}[a{i + 1}]"
        )
        filters.append(f)

    # Combine all speech clips
    mix_labels = "".join([f"[a{i + 1}]" for i in range(len(clips))])
    filters.append(f"{mix_labels}amix=inputs={len(clips)}:normalize=0[speech_raw]")

    # Sidechain setup
    filters.append("[speech_raw]asplit=2[speech_out][trigger]")
    filters.append("[0:a]aformat=sample_rates=48000:channel_layouts=stereo[bg_fixed]")
    filters.append("[bg_fixed][trigger]sidechaincompress=threshold=0.02:ratio=5:attack=50:release=600[bg_ducked]")
    filters.append("[bg_ducked][speech_out]amix=inputs=2:weights=1 1.5:normalize=0[out]")

    with open(filter_path, "w") as f:
        f.write(";".join(filters))

    run_cmd(
        ["ffmpeg"] + inputs + ["-filter_complex_script", filter_path, "-map", "[out]", "-c:a", "ac3", out, "-y"],
        "final mixing",
    )


def mux_video(v: str, a: str, lang: str, out: str, lang_name: str):
    """Combines video with new audio track."""
    title = f"AI - {lang_name}"
    cmd = [
        "ffmpeg",
        "-i",
        v,
        "-i",
        a,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "0:a",
        "-c:v",
        "copy",
        "-c:a",
        "ac3",
        "-metadata:s:a:0",
        f"language={lang}",
        "-metadata:s:a:0",
        f"title={title}",
        out,
        "-y",
    ]
    subprocess.run(cmd, capture_output=True)


def trim_silence(path: str):
    """Removes silence from start and end of audio file."""
    tmp = path + ".trim.wav"
    filter_chain = "areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse"
    cmd = ["ffmpeg", "-i", path, "-af", filter_chain, tmp, "-y"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            os.replace(tmp, path)
    except Exception:
        pass
    if os.path.exists(tmp):
        os.remove(tmp)


def extract_clean_segment(input_path: str, start: float, end: float, output_path: str):
    """Extracts and cleans audio segment for cloning."""
    duration = end - start
    filt = "highpass=f=100,afftdn=nf=-20,speechnorm=e=10:r=0.0001:l=1"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        input_path,
        "-af",
        filt,
        "-ac",
        "1",
        "-ar",
        "24000",
        output_path,
        "-y",
    ]
    subprocess.run(cmd, check=False, capture_output=True)
