import os
import glob
import logging
import shutil
import time
from typing import List, Dict, Tuple
from config import TEMP_DIR, MOCK_MODE
from infrastructure.ffmpeg import FFmpegWrapper
from core.gpu_manager import GPUManager

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

    # Dynamic GPU allocation
    # Demucs needs approx 3GB
    target_device = GPUManager.get_best_gpu(needed_mb=2000, purpose="Demucs")

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
        target_device,
        a_stereo,
    ]
    import subprocess

    # 10 minutes timeout for Demucs is reasonable for most videos.
    # If it takes longer, something is likely wrong or hardware is too slow.
    try:
        subprocess.run(demucs_cmd, check=True, timeout=600)
    except subprocess.TimeoutExpired:
        logger.error("Demucs separation TIMED OUT (600s).")
        raise RuntimeError("Audio separation took too long.")

    GPUManager.force_gc()

    found = glob.glob(os.path.join(TEMP_DIR, "**", "vocals.mp3"), recursive=True)
    if not found:
        raise FileNotFoundError("Demucs failed to produce vocals.mp3")
    return a_stereo, found[0]


def analyze_audio(vocals_path: str, diar_manager, whisper_manager) -> Tuple[List, List, Dict]:
    """Orchestrates diarization and transcription using provided managers with eager VRAM cleanup."""
    mpath = os.path.join(TEMP_DIR, "mono.wav")
    FFmpegWrapper.convert_audio(vocals_path, mpath, ac=1, ar=24000)

    durations = {}
    logger.info("Starting Audio Analysis (Stage 2)...")

    # 1. Diarization: Load, Process, Unload immediately
    t0 = time.perf_counter()
    logger.info("Stage 2a: Speaker Diarization...")
    diar_result = diar_manager.diarize(mpath)
    durations["2a. Diarization"] = time.perf_counter() - t0
    diar_manager.shutdown()
    GPUManager.force_gc()  # Aggressive cleanup

    # 2. Transcription: Load, Process, Unload immediately
    t0 = time.perf_counter()
    logger.info("Stage 2b: Transcription...")
    # WhisperManager.transcribe will auto-load if needed, but we can be explicit
    whisper_manager.load_model()
    trans_result = whisper_manager.transcribe(mpath)
    durations["2b. Transcription"] = time.perf_counter() - t0
    whisper_manager.shutdown()
    GPUManager.force_gc()  # Aggressive cleanup

    return diar_result, trans_result, durations


def mix_audio(bg: str, clips: List, out: str):
    """Mixes background audio with dubbed clips using sidechain compression and batching."""
    if not clips:
        shutil.copy(bg, out)
        return

    batch_size = 50
    intermediate_speech_files = []

    # 1. Process speech in batches to avoid OOM
    for b_idx in range(0, len(clips), batch_size):
        batch = clips[b_idx : b_idx + batch_size]
        batch_out = os.path.join(TEMP_DIR, f"speech_batch_{b_idx // batch_size}.wav")
        logger.info(f"Mixing speech batch {b_idx // batch_size + 1}/{(len(clips) - 1) // batch_size + 1}...")

        inputs = []
        filter_parts = []
        for i, (path, start, duration) in enumerate(batch):
            inputs.append(path)
            delay_ms = int(start * 1000)
            fade_st = max(0, duration - 0.05)
            # Process each clip: format, normalize, fade, and delay
            f = (
                f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-20:measured_TP=-1:measured_LRA=11:measured_thresh=-30:offset=0,"
                f"afade=t=in:st=0:d=0.05,afade=t=out:st={fade_st:.3f}:d=0.05,"
                f"adelay={delay_ms}|{delay_ms}[a{i}]"
            )
            filter_parts.append(f)

        mix_labels = "".join([f"[a{i}]" for i in range(len(batch))])
        filter_parts.append(f"{mix_labels}amix=inputs={len(batch)}:normalize=0[out_a]")

        filter_script_path = os.path.join(TEMP_DIR, f"batch_{b_idx // batch_size}_filter.txt")
        with open(filter_script_path, "w") as f_scr:
            f_scr.write(";".join(filter_parts))

        # Using raw subprocess here as batching needs custom output mapping (wav not ac3)
        cmd = ["ffmpeg"]
        for inp in inputs:
            cmd.extend(["-i", inp])
        cmd.extend(["-filter_complex_script", filter_script_path, "-map", "[out_a]", batch_out, "-y"])
        import subprocess

        subprocess.run(cmd, capture_output=True, check=True, timeout=300)

        intermediate_speech_files.append(batch_out)

    # 2. Combine all speech batches into one track
    speech_final = os.path.join(TEMP_DIR, "speech_final.wav")
    if len(intermediate_speech_files) > 1:
        logger.info("Merging speech batches...")
        merge_inputs = []
        for f in intermediate_speech_files:
            merge_inputs.extend(["-i", f])
        mix_labels = "".join([f"[{i}:a]" for i in range(len(intermediate_speech_files))])
        filter_str = f"{mix_labels}amix=inputs={len(intermediate_speech_files)}:normalize=0[out_a]"
        subprocess.run(
            ["ffmpeg"] + merge_inputs + ["-filter_complex", filter_str, "-map", "[out_a]", speech_final, "-y"],
            capture_output=True,
            check=True,
        )
    else:
        shutil.copy(intermediate_speech_files[0], speech_final)

    # 3. Final Master Mix with sidechain ducking and ghost track
    logger.info("Performing final master mix with sidechain...")
    master_filter_path = os.path.join(TEMP_DIR, "master_mix_filter.txt")

    # [0:a] is background, [1:a] is combined speech
    master_filters = [
        "[1:a]asplit=2[speech_out][trigger]",
        "[0:a]asplit=2[bg_main][bg_ghost_raw]",
        "[bg_main]aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-24:TP=-2:LRA=7,acompressor=threshold=-20dB:ratio=2:attack=20:release=200[bg_fixed]",
        "[bg_ghost_raw]aformat=sample_rates=48000:channel_layouts=stereo,lowpass=f=400,volume=0.05[bg_ghost]",
        "[bg_fixed][trigger]sidechaincompress=threshold=0.005:ratio=12:attack=30:release=600[bg_ducked]",
        "[bg_ducked][bg_ghost][speech_out]amix=inputs=3:weights=1 1 1:normalize=0,alimiter=limit=0.95[out]",
    ]

    with open(master_filter_path, "w") as f_master:
        f_master.write(";".join(master_filters))

    FFmpegWrapper.run_complex_script([bg, speech_final], master_filter_path, out)


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
