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


def test_centroids_follow_the_annotation_labels():
    from unittest.mock import MagicMock

    from core.diarization_engine import DiarizationManager

    annotation = MagicMock()
    annotation.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
    res = MagicMock(speaker_embeddings=[[0.5, 0.25], [1.0, 0.0]])
    assert DiarizationManager._centroids(res, annotation) == {"SPEAKER_00": [0.5, 0.25], "SPEAKER_01": [1.0, 0.0]}
    assert DiarizationManager._centroids(object(), annotation) == {}
