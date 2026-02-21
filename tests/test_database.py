import unittest
import os
from infrastructure.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_queue.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = Database(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_checkpointing(self):
        # Add a task
        self.db.add_task("/path/to/video.mp4", {"size": 100, "duration": 60})
        task = self.db.fetch_next_task()
        task_id = task["id"]

        # Initially no result
        res = self.db.get_step_result(task_id, "step1")
        self.assertIsNone(res)

        # Save result
        test_data = {"key": "value", "list": [1, 2, 3]}
        self.db.save_step_result(task_id, "step1", "DONE", result_data=test_data)

        # Retrieve result
        res = self.db.get_step_result(task_id, "step1")
        self.assertEqual(res, test_data)

        # Update to Error
        self.db.save_step_result(task_id, "step1", "FAILED", error_msg="Something went wrong")
        res = self.db.get_step_result(task_id, "step1")
        self.assertIsNone(res)  # Should be None because status is not DONE

    def test_task_management(self):
        path = "/test/video.mkv"
        meta = {"size": 1024, "duration": 30.5, "target_langs": "pl,en"}

        # 1. Add task
        status = self.db.add_task(path, meta)
        self.assertEqual(status, "queued")

        # 2. Add duplicate task (should be ignored if QUEUED)
        status = self.db.add_task(path, meta)
        self.assertEqual(status, "ignored")

        # 3. Fetch task
        task = self.db.fetch_next_task()
        self.assertIsNotNone(task)
        self.assertEqual(task["path"], path)
        # Note: fetch_next_task returns the row BEFORE updating status to PROCESSING
        self.assertEqual(task["status"], "QUEUED")

        # Verify it really updated in DB
        all_tasks = self.db.get_all_tasks()
        self.assertEqual(all_tasks[0]["status"], "PROCESSING")

        # 4. Update status
        self.db.update_status(task["id"], "DONE")
        all_tasks = self.db.get_all_tasks()
        self.assertEqual(all_tasks[0]["status"], "DONE")

        # 5. Re-add DONE task (should be re-queued and cache cleared)
        self.db.save_step_result(task["id"], "Stage 1", "DONE", {"res": 1})
        status = self.db.add_task(path, meta)
        self.assertEqual(status, "re-queued")
        res = self.db.get_step_result(task["id"], "Stage 1")
        self.assertIsNone(res)

    def test_queue_stats_and_reset(self):
        self.db.add_task("v1.mp4", {})
        self.db.add_task("v2.mp4", {})
        self.db.add_task("v3.mp4", {})

        # Start one
        self.db.fetch_next_task()

        stats = self.db.get_queue_stats()
        self.assertEqual(stats["QUEUED"], 2)
        self.assertEqual(stats["PROCESSING"], 1)

        # Reset interrupted
        count = self.db.reset_interrupted_tasks()
        self.assertEqual(count, 1)
        stats = self.db.get_queue_stats()
        self.assertEqual(stats["QUEUED"], 3)
        self.assertNotIn("PROCESSING", stats)

    def test_progress_calculation(self):
        self.db.add_task("prog.mp4", {})
        task = self.db.fetch_next_task()
        task_id = task["id"]

        # Initial processing state (Stage 1 implicitly started)
        self.assertEqual(self.db.get_task_progress(task_id), 1)

        # Add some steps
        self.db.save_step_result(task_id, "Stage 1: Audio Separation", "DONE", {})
        self.assertEqual(self.db.get_task_progress(task_id), 1)

        self.db.save_step_result(task_id, "Stage 2: Audio Analysis", "PENDING", {})
        self.assertEqual(self.db.get_task_progress(task_id), 2)

        self.db.save_step_result(task_id, "Stage 2: Audio Analysis", "DONE", {})
        self.assertEqual(self.db.get_task_progress(task_id), 2)


if __name__ == "__main__":
    unittest.main()
