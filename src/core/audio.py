import os
import glob
import logging
import gc
import shutil
import time
from typing import List, Dict, Tuple
from config import DEVICE_AUDIO, TEMP_DIR, WHISPER_MODEL, MOCK_MODE
from infrastructure.ffmpeg import FFmpegWrapper

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def prep_audio(vpath: str) -> Tuple[str, str]:
    """Extracts original stereo audio and separates vocals using Demucs. Mocks in MOCK_MODE."""
    a_stereo = os.path.join(TEMP_DIR, "orig.wav")
    FFmpegWrapper.extract_audio(vpath, a_stereo)

    if MOCK_MODE:
        logger.info("Demucs: MOCK_MODE enabled. Using original audio as vocals.")
        vocals_path = os.path.join(TEMP_DIR, "vocals.mp3")
        shutil.copy(a_stereo, vocals_path)
        return a_stereo, vocals_path

    demucs_cmd = [
        "demucs",
        "--mp3",
        "--two-stems",
        "vocals",
        "-o",
        TEMP_DIR,
        "-n",
        "htdemucs_ft",
        "--device",
        DEVICE_AUDIO,
        a_stereo,
    ]
    import subprocess

    subprocess.run(demucs_cmd, check=True)

    found = glob.glob(os.path.join(TEMP_DIR, "**", "vocals.mp3"), recursive=True)
    if not found:
        raise FileNotFoundError("Demucs failed to produce vocals.mp3")
    return a_stereo, found[0]


def run_diarization(mpath: str) -> List[Dict]:
    if MOCK_MODE:
        return [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]

    if not torch:
        raise ImportError("Torch is required for diarization but not installed.")

    from pyannote.audio import Pipeline

    p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ.get("HF_TOKEN"))
    p.to(torch.device(DEVICE_AUDIO))
    res = p(mpath)
    annotation = getattr(res, "speaker_diarization", getattr(res, "diarization", getattr(res, "annotation", res)))
    diar_result = [
        {"start": s.start, "end": s.end, "speaker": label} for s, _, label in annotation.itertracks(yield_label=True)
    ]
    del p
    gc.collect()
    if "cuda" in DEVICE_AUDIO:
        torch.cuda.empty_cache()
    return diar_result


def run_transcription(mpath: str) -> List[Dict]:
    if MOCK_MODE:
        return [{"start": 0.0, "end": 5.0, "text": "Mock transcription", "avg_logprob": -0.1, "no_speech_prob": 0.01}]

    if not torch:
        # Faster-whisper might work without torch on CPU if using int8, but usually it needs ctranslate2/torch
        pass

    from faster_whisper import WhisperModel

    device = "cuda" if "cuda" in DEVICE_AUDIO else "cpu"
    device_index = int(DEVICE_AUDIO.split(":")[-1]) if "cuda" in DEVICE_AUDIO else 0
    m = WhisperModel(
        WHISPER_MODEL, device=device, device_index=device_index, compute_type="float16" if device == "cuda" else "int8"
    )
    ts, _ = m.transcribe(mpath)
    res = [
        {
            "start": x.start,
            "end": x.end,
            "text": x.text.strip(),
            "avg_logprob": x.avg_logprob,
            "no_speech_prob": x.no_speech_prob,
        }
        for x in ts
        if x.avg_logprob >= -1.0
    ]
    del m
    gc.collect()
    if torch and "cuda" in DEVICE_AUDIO:
        torch.cuda.empty_cache()
    return res


def analyze_audio(vocals_path: str) -> Tuple[List, List, Dict]:
    """Orchestrates diarization and transcription."""
    mpath = os.path.join(TEMP_DIR, "mono.wav")
    FFmpegWrapper.convert_audio(vocals_path, mpath, ac=1, ar=24000)

    durations = {}
    logging.info(f"Starting Audio Analysis on {DEVICE_AUDIO}...")

    t0 = time.perf_counter()
    diar_result = run_diarization(mpath)
    durations["2a. Diarization"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    trans_result = run_transcription(mpath)
    durations["2b. Transcription"] = time.perf_counter() - t0

    return diar_result, trans_result, durations


def mix_audio(bg: str, clips: List, out: str):
    if not clips:
        shutil.copy(bg, out)
        return

    filter_path = os.path.join(TEMP_DIR, "mix_filter.txt")
    inputs = [bg]
    filters = []

    for i, (path, start, duration) in enumerate(clips):
        inputs.append(path)
        delay_ms = int(start * 1000)
        fade_st = max(0, duration - 0.05)
        f = (
            f"[{i + 1}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-20:measured_TP=-1:measured_LRA=11:measured_thresh=-30:offset=0,"
            f"afade=t=in:st=0:d=0.05,afade=t=out:st={fade_st:.3f}:d=0.05,"
            f"adelay={delay_ms}|{delay_ms}[a{i + 1}]"
        )
        filters.append(f)

    mix_labels = "".join([f"[a{i + 1}]" for i in range(len(clips))])
    filters.append(f"{mix_labels}amix=inputs={len(clips)}:normalize=0[speech_raw]")
    filters.append("[0:a]asplit=2[bg_main][bg_ghost_raw]")
    filters.append(
        "[bg_main]aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-24:TP=-2:LRA=7,acompressor=threshold=-20dB:ratio=2:attack=20:release=200[bg_fixed]"
    )
    filters.append("[bg_ghost_raw]aformat=sample_rates=48000:channel_layouts=stereo,lowpass=f=400,volume=0.05[bg_ghost]")
    filters.append("[speech_raw]asplit=2[speech_out][trigger]")
    filters.append("[bg_fixed][trigger]sidechaincompress=threshold=0.005:ratio=12:attack=30:release=600[bg_ducked]")
    filters.append("[bg_ducked][bg_ghost][speech_out]amix=inputs=3:weights=1 1 1:normalize=0,alimiter=limit=0.95[out]")

    with open(filter_path, "w") as f:
        f.write(";".join(filters))
    FFmpegWrapper.run_complex_script(inputs, filter_path, out)


def get_audio_languages(vpath: str) -> List[str]:
    """Returns a list of language codes for audio streams in the video."""
    try:
        meta = FFmpegWrapper.get_metadata(vpath)
        langs = []
        for stream in meta.get("streams", []):
            if stream.get("codec_type") == "audio":
                langs.append(stream.get("tags", {}).get("language", "und").lower())
        return langs
    except Exception:
        return []


def extract_clean_segment(input_path: str, start: float, end: float, output_path: str):
    """Extracts and cleans audio segment for cloning."""
    duration = end - start
    filt = "highpass=f=100,afftdn=nf=-20,speechnorm=e=10:r=0.0001:l=1"
    # Using raw subprocess here as FFmpegWrapper.apply_filter takes full input,
    # but here we need seeking (-ss). Could extend Wrapper later.
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
    import subprocess

    subprocess.run(cmd, check=False, capture_output=True)


def trim_and_pad_silence(path: str, target_dur: float):
    """
    Intelligently trims leading/trailing silence and then pads or slightly
    adjusts speed to match the target duration exactly.
    """
    tmp = path + ".proc.wav"
    # 1. Trim silence from both ends
    trim_filt = "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_silence=0.05:start_threshold=-50dB,areverse"

    # Run trimming
    import subprocess

    subprocess.run(["ffmpeg", "-i", path, "-af", trim_filt, tmp, "-y"], capture_output=True)

    if not os.path.exists(tmp) or os.path.getsize(tmp) < 100:
        return  # Keep original if trimming failed

    # 2. Check new duration
    current_dur = FFmpegWrapper.get_duration(tmp)

    # 3. Decision: Pad if too short, stretch if too long (but only slightly)
    final_filt = []
    if current_dur < target_dur:
        pad_dur = target_dur - current_dur
        final_filt.append(f"apad=pad_dur={pad_dur:.3f}")
    elif current_dur > target_dur:
        speed = current_dur / target_dur
        if speed <= 1.05:
            final_filt.append(f"atempo={speed:.3f}")

    if final_filt:
        tmp2 = tmp + ".final.wav"
        subprocess.run(["ffmpeg", "-i", tmp, "-af", ",".join(final_filt), tmp2, "-y"], capture_output=True)
        if os.path.exists(tmp2):
            os.replace(tmp2, path)
    else:
        os.replace(tmp, path)

    if os.path.exists(tmp):
        os.remove(tmp)
