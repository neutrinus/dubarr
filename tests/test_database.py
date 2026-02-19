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


if __name__ == "__main__":
    unittest.main()
