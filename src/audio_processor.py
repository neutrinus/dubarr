import os
import glob
import subprocess
import logging
import torch
import gc
import shutil
import time
from typing import List, Dict, Tuple
from config import DEVICE_AUDIO, TEMP_DIR, WHISPER_MODEL, MOCK_MODE
from utils import run_cmd


def prep_audio(vpath: str) -> Tuple[str, str]:
    """Extracts original stereo audio and separates vocals using Demucs. Mocks in MOCK_MODE."""
    a_stereo = os.path.join(TEMP_DIR, "orig.wav")
    run_cmd(["ffmpeg", "-i", vpath, "-vn", "-ac", "2", "-y", a_stereo], "extract audio")

    if MOCK_MODE:
        logging.info("Demucs: MOCK_MODE enabled. Using original audio as vocals.")
        vocals_path = os.path.join(TEMP_DIR, "vocals.mp3")
        run_cmd(["ffmpeg", "-i", a_stereo, "-y", vocals_path], "mock separation")
        return a_stereo, vocals_path

    # Demucs expects a device string like "cuda" or "cpu"
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
    run_cmd(demucs_cmd, "demucs separation")

    found = glob.glob(os.path.join(TEMP_DIR, "**", "vocals.mp3"), recursive=True)
    if not found:
        raise FileNotFoundError("Demucs failed to produce vocals.mp3")
    return a_stereo, found[0]


def run_diarization(mpath: str) -> List[Dict]:
    """Runs speaker diarization using Pyannote. Mocks in MOCK_MODE."""
    if os.environ.get("MOCK_MODE") == "1":
        return [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]

    from pyannote.audio import Pipeline

    logging.info(f"Diarization: Loading on {DEVICE_AUDIO}")

    p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ.get("HF_TOKEN"))
    p.to(torch.device(DEVICE_AUDIO))
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
    if "cuda" in DEVICE_AUDIO:
        torch.cuda.empty_cache()
    return diar_result


def run_transcription(mpath: str) -> List[Dict]:
    """Runs transcription using Faster-Whisper. Mocks in MOCK_MODE."""
    if MOCK_MODE:
        return [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a mock transcription for testing.",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.01,
            }
        ]

    from faster_whisper import WhisperModel

    logging.info(f"Whisper: Loading on {DEVICE_AUDIO}")

    # Parse device string for Faster-Whisper
    if "cuda" in DEVICE_AUDIO:
        device = "cuda"
        device_index = int(DEVICE_AUDIO.split(":")[-1])
        compute_type = "float16"
    else:
        device = "cpu"
        device_index = 0
        compute_type = "int8"  # CPU works better/faster with int8 quantized

    m = WhisperModel(WHISPER_MODEL, device=device, device_index=device_index, compute_type=compute_type)
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
    if "cuda" in DEVICE_AUDIO:
        torch.cuda.empty_cache()
    return trans_result


def analyze_audio(vocals_path: str) -> Tuple[List, List, Dict]:
    """Orchestrates diarization and transcription."""
    mpath = os.path.join(TEMP_DIR, "mono.wav")
    run_cmd(["ffmpeg", "-i", vocals_path, "-ac", "1", mpath, "-y"], "mono conversion")

    durations = {}
    logging.info(f"Starting Sequential Audio Analysis on {DEVICE_AUDIO}...")

    t0 = time.perf_counter()
    diar_result = run_diarization(mpath)
    durations["2a. Diarization"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    trans_result = run_transcription(mpath)
    durations["2b. Transcription"] = time.perf_counter() - t0

    return diar_result, trans_result, durations


def get_audio_languages(vpath: str) -> List[str]:
    """Returns a list of language codes for audio streams in the video."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index:stream_tags=language",
        "-of",
        "csv=p=0",
        vpath,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        langs = []
        for line in res.stdout.strip().split("\n"):
            if "," in line:
                # Format is: index,lang (e.g. 1,pol)
                parts = line.split(",")
                if len(parts) >= 2:
                    langs.append(parts[1].lower())
        return langs
    except Exception as e:
        logging.warning(f"Failed to detect audio languages: {e}")
        return []


def mix_audio(bg: str, clips: List, out: str):
    """Mixes background audio with dubbed clips using aggressive sidechain and ambient ghosting."""
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

        # Format each clip: Stereo 48k, fade in/out, delay, and normalization
        f = (
            f"[{i + 1}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-20:measured_TP=-1:measured_LRA=11:measured_thresh=-30:offset=0,"
            f"afade=t=in:st=0:d=0.05,afade=t=out:st={fade_st:.3f}:d=0.05,"
            f"adelay={delay_ms}|{delay_ms}[a{i + 1}]"
        )
        filters.append(f)

    # Combine all speech clips
    mix_labels = "".join([f"[a{i + 1}]" for i in range(len(clips))])
    filters.append(f"{mix_labels}amix=inputs={len(clips)}:normalize=0[speech_raw]")

    # Split original audio into main and ambient ghost
    filters.append("[0:a]asplit=2[bg_main][bg_ghost_raw]")

    # Process main background: normalize and compress
    filters.append(
        "[bg_main]aformat=sample_rates=48000:channel_layouts=stereo,"
        "loudnorm=I=-24:TP=-2:LRA=7,"
        "acompressor=threshold=-20dB:ratio=2:attack=20:release=200[bg_fixed]"
    )

    # Process ghost background: Low-pass and very quiet to keep acoustics during ducking
    filters.append("[bg_ghost_raw]aformat=sample_rates=48000:channel_layouts=stereo,lowpass=f=400,volume=0.05[bg_ghost]")

    # Aggressive sidechain setup (ratio 12 instead of 4)
    filters.append("[speech_raw]asplit=2[speech_out][trigger]")
    filters.append("[bg_fixed][trigger]sidechaincompress=threshold=0.005:ratio=12:attack=30:release=600[bg_ducked]")

    # Final mix: Ducked Main + Constant Ambient Ghost + AI Speech
    filters.append("[bg_ducked][bg_ghost][speech_out]amix=inputs=3:weights=1 1 1:normalize=0,alimiter=limit=0.95[out]")

    with open(filter_path, "w") as f:
        f.write(";".join(filters))

    run_cmd(
        ["ffmpeg"] + inputs + ["-filter_complex_script", filter_path, "-map", "[out]", "-c:a", "ac3", out, "-y"],
        "final mixing",
    )


def mux_video(v: str, audio_tracks: List[Tuple[str, str, str]], out: str):
    """
    Combines video with multiple new audio tracks.
    audio_tracks: List of (audio_path, lang_code, lang_name)
    """
    cmd = ["ffmpeg", "-i", v]
    for a_path, _, _ in audio_tracks:
        cmd.extend(["-i", a_path])

    # Map video from source
    cmd.extend(["-map", "0:v"])

    # Map all new audio tracks first
    for i in range(len(audio_tracks)):
        cmd.extend(["-map", f"{i + 1}:a"])

    # Map original audio tracks from source
    cmd.extend(["-map", "0:a"])

    # Set metadata for each new audio track
    for i, (_, lang, lang_name) in enumerate(audio_tracks):
        title = f"AI - {lang_name}"
        cmd.extend([f"-metadata:s:a:{i}", f"language={lang}", f"-metadata:s:a:{i}", f"title={title}"])

    cmd.extend(["-c:v", "copy", "-c:a", "ac3", out, "-y"])

    logging.info(f"Muxing {len(audio_tracks)} tracks into {out}")
    subprocess.run(cmd, capture_output=True)


def trim_and_pad_silence(path: str, target_dur: float):
    """
    Intelligently trims leading/trailing silence and then pads or slightly
    adjusts speed to match the target duration exactly.
    """
    tmp = path + ".proc.wav"
    # 1. Trim silence from both ends
    trim_filt = "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_silence=0.05:start_threshold=-50dB,areverse"

    # Run trimming
    subprocess.run(["ffmpeg", "-i", path, "-af", trim_filt, tmp, "-y"], capture_output=True)

    if not os.path.exists(tmp) or os.path.getsize(tmp) < 100:
        return  # Keep original if trimming failed

    # 2. Check new duration
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", tmp]
    try:
        current_dur = float(subprocess.check_output(cmd).strip())
    except Exception:
        return

    # 3. Decision: Pad if too short, stretch if too long (but only slightly)
    # If it's way too long, atempo will handle it in _apply_mastering_and_speed,
    # but here we handle fine-tuning.

    final_filt = []
    if current_dur < target_dur:
        # Pad with silence at the end to match exactly
        pad_dur = target_dur - current_dur
        final_filt.append(f"apad=pad_dur={pad_dur:.3f}")
    elif current_dur > target_dur:
        # If it's only slightly longer, we can speed it up here
        speed = current_dur / target_dur
        if speed <= 1.05:  # Only if tiny change needed
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
