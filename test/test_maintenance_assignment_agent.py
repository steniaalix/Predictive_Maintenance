import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from database import maintenance_database as db
from maintenance_assignment_agent import create_maintenance_assignment

class TestMaintenanceAssignmentAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "maintenance_tasks.db"
        self.path_patch = patch.object(db, "DATABASE_PATH", self.db_path)
        self.path_patch.start()
        db.create_database()
        self.writer_patch = patch("maintenance_assignment_agent.generate_task_description",
                                  return_value="Inspect the supplied operating conditions.")
        self.mock_writer = self.writer_patch.start()
    def tearDown(self):
        self.writer_patch.stop()
        self.path_patch.stop()
        self.temp_dir.cleanup()
    def test_creates_task(self):
        result = create_maintenance_assignment("TEST_MACHINE", "CRITICAL", "PWF", 0.94)
        self.assertTrue(result["task_created"])
        self.assertFalse(result["duplicate"])
        self.assertEqual(result["task"]["assigned_role"], "Equipment Maintenance Team")
        self.assertEqual(result["task"]["priority"], "HIGH")
        self.assertEqual(result["task"]["status"], "PENDING")
        self.mock_writer.assert_called_once()
    def test_duplicate_active_incident_reuses_task(self):
        first = create_maintenance_assignment("TEST_MACHINE", "CRITICAL", "PWF", 0.94)
        second = create_maintenance_assignment("TEST_MACHINE", "CRITICAL", "PWF", 0.96)
        self.assertTrue(first["task_created"])
        self.assertFalse(second["task_created"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertEqual(len(db.get_all_tasks()), 1)
        self.mock_writer.assert_called_once()
    def test_normal_does_not_create_task(self):
        result = create_maintenance_assignment("NORMAL_TEST_MACHINE", "NORMAL")
        self.assertFalse(result["task_created"])
        self.assertIsNone(result["task_id"])
        self.assertEqual(db.get_all_tasks(), [])
        self.mock_writer.assert_not_called()
    def test_completed_task_allows_new_incident_task(self):
        first = create_maintenance_assignment("TEST_MACHINE", "CRITICAL", "PWF", 0.94)
        db.update_task_status(first["task_id"], "IN_PROGRESS")
        db.update_task_status(first["task_id"],"COMPLETED")
        second = create_maintenance_assignment("TEST_MACHINE", "CRITICAL", "PWF", 0.91)
        self.assertTrue(second["task_created"])
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertEqual(len(db.get_all_tasks()), 2)
    def test_severity_escalation_creates_separate_task(self):
        warning = create_maintenance_assignment("TEST_MACHINE", "WARNING", "PWF", 0.60)
        critical = create_maintenance_assignment("TEST_MACHINE", "CRITICAL", "PWF", 0.94)
        self.assertTrue(warning["task_created"])
        self.assertTrue(critical["task_created"])
        self.assertNotEqual(warning["task_id"], critical["task_id"])

if __name__ == "__main__":
    unittest.main()
