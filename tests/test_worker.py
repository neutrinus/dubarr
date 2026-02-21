import unittest
from unittest.mock import MagicMock
from core.worker import JobWorker


class TestJobWorker(unittest.TestCase):
    def test_worker_flow(self):
        mock_db = MagicMock()
        mock_pipeline = MagicMock()

        # 1. First call returns a task
        # 2. Second call sets stop_event to exit loop
        task = {"id": 1, "path": "video.mp4"}
        mock_db.fetch_next_task.side_effect = [task, None]

        worker = JobWorker(mock_db, lambda: mock_pipeline)

        # Mock stop_event to exit after one task
        def stop_after_fetch(*args):
            worker.stop()
            return task

        mock_db.fetch_next_task.side_effect = stop_after_fetch

        worker._run()

        mock_pipeline.process_video.assert_called_once_with("video.mp4", task_id=1)
        mock_db.update_status.assert_called_once_with(1, "DONE")

    def test_worker_failure(self):
        mock_db = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = Exception("Boom")

        task = {"id": 1, "path": "fail.mp4"}
        worker = JobWorker(mock_db, lambda: mock_pipeline)

        def stop_after_fetch(*args):
            worker.stop()
            return task

        mock_db.fetch_next_task.side_effect = stop_after_fetch

        worker._run()

        mock_db.update_status.assert_called_once_with(1, "FAILED")
