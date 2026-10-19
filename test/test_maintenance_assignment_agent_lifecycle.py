import unittest
from uuid import uuid4

from maintenance_assignment_agent import create_maintenance_assignment
from database.maintenance_database import (
    get_task,
    get_all_tasks,
    update_task_status,
)


class TestMaintenanceAssignmentAgentLifecycle(unittest.TestCase):

    def setUp(self):
        # Unique machine ID prevents interference from earlier tests.
        self.machine_id = f"agent_lifecycle_{uuid4().hex}"

    def test_new_incident_creates_task(self):
        result = create_maintenance_assignment(
            machine_id=self.machine_id,
            severity="CRITICAL",
            failure_type="PWF",
            failure_probability=0.95,
            anomaly=False,
        )

        self.assertTrue(result["task_created"])
        self.assertFalse(result["duplicate"])
        self.assertIsNotNone(result["task_id"])
        self.assertEqual(result["task"]["status"], "PENDING")

        # Verify that the task was persisted in SQLite.
        saved_task = get_task(result["task_id"])
        self.assertIsNotNone(saved_task)
        self.assertEqual(saved_task["machine_id"], self.machine_id)

    def test_duplicate_incident_does_not_create_another_task(self):
        first_result = create_maintenance_assignment(
            machine_id=self.machine_id,
            severity="CRITICAL",
            failure_type="PWF",
            failure_probability=0.95,
            anomaly=False,
        )

        second_result = create_maintenance_assignment(
            machine_id=self.machine_id,
            severity="CRITICAL",
            failure_type="PWF",
            failure_probability=0.95,
            anomaly=False,
        )

        self.assertTrue(first_result["task_created"])
        self.assertFalse(second_result["task_created"])
        self.assertTrue(second_result["duplicate"])

        # Both results should refer to the same task.
        self.assertEqual(
            first_result["task_id"],
            second_result["task_id"],
        )

    def test_normal_severity_creates_no_task(self):
        result = create_maintenance_assignment(
            machine_id=self.machine_id,
            severity="NORMAL",
            failure_type=None,
            failure_probability=None,
            anomaly=False,
        )

        self.assertFalse(result["task_created"])
        self.assertFalse(result["duplicate"])
        self.assertIsNone(result["task_id"])
        self.assertIsNone(result["task"])

    def test_completed_task_allows_new_task_for_same_incident(self):
        first_result = create_maintenance_assignment(
            machine_id=self.machine_id,
            severity="CRITICAL",
            failure_type="PWF",
            failure_probability=0.95,
            anomaly=False,
        )

        self.assertTrue(first_result["task_created"])

        update_task_status(first_result["task_id"], "IN_PROGRESS")
        update_task_status(first_result["task_id"], "COMPLETED")

        second_result = create_maintenance_assignment(
            machine_id=self.machine_id,
            severity="CRITICAL",
            failure_type="PWF",
            failure_probability=0.95,
            anomaly=False,
        )

        self.assertTrue(second_result["task_created"])
        self.assertFalse(second_result["duplicate"])
        self.assertNotEqual(
            first_result["task_id"],
            second_result["task_id"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)