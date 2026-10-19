import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from database import maintenance_database as db

class TestMaintenanceDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "maintenance_tasks.db"
        self.path_patch = patch.object(db, "DATABASE_PATH", self.db_path)
        self.path_patch.start()
        db.create_database()
    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()
    def make_task(self):
        return db.create_task("TEST_MACHINE", "CRITICAL", "PWF", 0.94,
            "Equipment Maintenance Team", "HIGH", "Inspect promptly", "Test task description.")
    def test_create_and_get_task(self):
        task = db.get_task(self.make_task())
        self.assertEqual(task["machine_id"], "TEST_MACHINE")
        self.assertEqual(task["status"], "PENDING")
    def test_get_all_tasks(self):
        self.make_task()
        self.assertEqual(len(db.get_all_tasks()), 1)
    def test_find_active_task(self):
        task_id = self.make_task()
        found = db.get_active_task_for_incident("TEST_MACHINE", "CRITICAL", "PWF")
        self.assertEqual(found["task_id"], task_id)
    def test_completed_task_is_not_active(self):
        task_id = self.make_task()
        self.assertTrue(db.update_task_status(task_id, "IN_PROGRESS"))
        self.assertTrue(db.update_task_status(task_id, "COMPLETED"))
        self.assertIsNone(db.get_active_task_for_incident("TEST_MACHINE", "CRITICAL", "PWF"))
    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            db.update_task_status(1, "BROKEN")

if __name__ == "__main__":
    unittest.main()
