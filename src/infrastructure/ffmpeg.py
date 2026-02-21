import subprocess
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


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
    def get_duration(path: str) -> float:
        """Gets duration of a media file."""
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        return float(subprocess.check_output(cmd, timeout=10).strip())

    @staticmethod
    def apply_filter(input_path: str, output_path: str, filter_chain: str):
        """Applies a generic audio filter chain."""
        cmd = ["ffmpeg", "-i", input_path, "-af", filter_chain, output_path, "-y"]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)

    @staticmethod
    def mux_video(video_path: str, audio_tracks: List[Tuple[str, str, str]], output_path: str):
        """Muxes multiple audio tracks into a video container."""
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

        cmd.extend(["-c:v", "copy", "-c:a", "ac3", output_path, "-y"])
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)

    @staticmethod
    def run_complex_script(inputs: List[str], script_path: str, output_path: str):
        """Runs ffmpeg with a complex filter script."""
        cmd = ["ffmpeg"]
        for inp in inputs:
            cmd.extend(["-i", inp])
        cmd.extend(["-filter_complex_script", script_path, "-map", "[out]", "-c:a", "ac3", output_path, "-y"])
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
