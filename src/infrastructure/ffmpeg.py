import subprocess
import json
import logging
import re
from typing import List, Dict, Optional, Tuple

from config import AUDIO_BITRATE, AUDIO_CODEC

logger = logging.getLogger(__name__)

# ffmpeg encoder per AUDIO_CODEC value. AAC plays everywhere (browsers, Android, TVs); AC3/E-AC3 suit AV receivers.
AUDIO_ENCODERS = {"aac": "aac", "ac3": "ac3", "eac3": "eac3", "opus": "libopus"}


def audio_codec_args(codec: str, bitrate: str, stream: str = "") -> List[str]:
    """ffmpeg arguments that encode audio stream(s) `stream` (e.g. ":0") with the configured codec."""
    encoder = AUDIO_ENCODERS.get(codec.lower(), "aac")
    return [f"-c:a{stream}", encoder, f"-b:a{stream}", bitrate]


class FFmpegWrapper:
    """Encapsulates all FFmpeg operations for the project."""

    @staticmethod
    def get_metadata(path: str) -> Dict:
        """Extracts metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index:stream_tags=language:stream=codec_type",
            "-of",
            "json",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return json.loads(res.stdout)

    @staticmethod
    def extract_audio(input_path: str, output_path: str, channels: int = 2):
        """Extracts audio from video to a WAV file."""
        cmd = ["ffmpeg", "-i", input_path, "-vn", "-ac", str(channels), "-y", output_path]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extract_audio failed: {e.stderr}")
            raise e

    @staticmethod
    def convert_audio(input_path: str, output_path: str, ac: int = 1, ar: int = 24000):
        """Converts audio format (e.g., for TTS or Diarization)."""
        cmd = ["ffmpeg", "-i", input_path, "-ac", str(ac), "-ar", str(ar), output_path, "-y"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg convert_audio failed: {e.stderr}")
            raise e

    @staticmethod
    def mean_volume(path: str, start: float, duration: float) -> Optional[float]:
        """Mean level (dB) of a span of an audio file, None when it cannot be measured."""
        cmd = ["ffmpeg", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", path, "-af", "volumedetect", "-f", "null", "-"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stderr
        except subprocess.TimeoutExpired:
            return None
        match = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", out)
        if not match or match.group(1) == "-inf":
            return None
        return float(match.group(1))

    @staticmethod
    def get_duration(path: str) -> float:
        """Gets duration of a media file."""
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        try:
            return float(subprocess.check_output(cmd, timeout=10).strip())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            logger.error(f"FFmpeg get_duration failed or timed out for {path}")
            return 0.0

    @staticmethod
    def apply_filter(input_path: str, output_path: str, filter_chain: str):
        """Applies a generic audio filter chain."""
        cmd = ["ffmpeg", "-i", input_path, "-af", filter_chain, output_path, "-y"]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)

    @staticmethod
    def mux_video(
        video_path: str, audio_tracks: List[Tuple[str, str, str]], output_path: str, codec: str = "", bitrate: str = ""
    ):
        """Adds the dubbed tracks (encoded once, with `codec`) in front of the original audio, which is copied as is."""
        cmd = ["ffmpeg", "-i", video_path]
        for a_path, _, _ in audio_tracks:
            cmd.extend(["-i", a_path])

        cmd.extend(["-map", "0:v"])
        for i in range(len(audio_tracks)):
            cmd.extend(["-map", f"{i + 1}:a"])
        cmd.extend(["-map", "0:a"])

        for i, (_, lang, lang_name) in enumerate(audio_tracks):
            title = f"AI - {lang_name}"
            cmd.extend([f"-metadata:s:a:{i}", f"language={lang}", f"-metadata:s:a:{i}", f"title={title}"])

        cmd.extend(["-c:v", "copy", "-c:a", "copy"])
        for i in range(len(audio_tracks)):
            cmd.extend(audio_codec_args(codec or AUDIO_CODEC, bitrate or AUDIO_BITRATE, f":{i}"))
        cmd.extend([output_path, "-y"])
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)

    @staticmethod
    def run_complex_script(inputs: List[str], script_path: str, output_path: str):
        """Runs ffmpeg with a complex filter script."""
        cmd = ["ffmpeg"]
        for inp in inputs:
            cmd.extend(["-i", inp])
        # The output format follows the extension (the final mix is a lossless WAV, encoded once when muxing)
        cmd.extend(["-filter_complex_script", script_path, "-map", "[out]", output_path, "-y"])
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
