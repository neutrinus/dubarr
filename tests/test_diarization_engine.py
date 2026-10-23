import unittest
from unittest.mock import MagicMock, patch

from core.diarization_engine import DiarizationManager


class TestDiarizationManager(unittest.TestCase):
    def test_passes_preloaded_waveform_and_maps_tracks(self):
        manager = DiarizationManager(inference_lock=None)
        segment = MagicMock(start=1.0, end=2.5)
        annotation = MagicMock()
        annotation.itertracks.return_value = [(segment, None, "SPEAKER_01")]
        manager.pipeline = MagicMock(return_value=MagicMock(speaker_diarization=annotation))
        audio = {"waveform": "tensor", "sample_rate": 24000}

        with patch.object(DiarizationManager, "_load_waveform", return_value=audio) as load:
            result = manager._run_diarize("mono.wav")

        load.assert_called_once_with("mono.wav")
        manager.pipeline.assert_called_once_with(audio)
        self.assertEqual(result, [{"start": 1.0, "end": 2.5, "speaker": "SPEAKER_01"}])


if __name__ == "__main__":
    unittest.main()
