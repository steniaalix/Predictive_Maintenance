import unittest

from database.maintenance_database import (
    create_task,
    get_task,
    get_all_tasks,
    get_active_task_for_incident,
    update_task_status,
)

class TestMaintenanceLifecycle(unittest.TestCase):
    @classmethod

    def setUpClass(cls):
        from database.maintenance_database import create_database

        create_database()

    def setUp(self):
        self.machine_id=f"lifecycle_test_{id(self)}"

        self.task_id=create_task(
            machine_id=self.machine_id,
            severity="CRITICAL",
            failure_type="PWF",
            failure_probability=0.95,
            assigned_role="Equipment Maintenance Team",
            priority="High",
            response_target="Inspect promptly",
            task_description="Inspect power-related operating conditions.",   
        )

    def test_task_is_created_and_retrieved(self):
        task = get_task(self.task_id)

        self.assertIsNotNone(task)
        self.assertEqual(task["task_id"], self.task_id)
        self.assertEqual(task["machine_id"], self.machine_id)
        self.assertEqual(task["status"], "PENDING")


    def test_all_tasks_returns_a_list(self):
        tasks = get_all_tasks()

        self.assertIsInstance(tasks, list)
        self.assertTrue(
            any(task["task_id"] == self.task_id for task in tasks)
        )


    def test_active_task_is_found(self):
        task = get_active_task_for_incident(
            self.machine_id, "CRITICAL", "PWF"
        )

        self.assertIsNotNone(task)
        self.assertEqual(task["task_id"], self.task_id)

    def test_status_changes_to_in_progress(self):
        updated = update_task_status(self.task_id, "IN_PROGRESS")

        self.assertTrue(updated)
        self.assertEqual(
            get_task(self.task_id)["status"], "IN_PROGRESS"
        )

    def test_completed_task_is_no_longer_active(self):
        update_task_status(self.task_id, "IN_PROGRESS")
        update_task_status(self.task_id, "COMPLETED")

        active_task=get_active_task_for_incident(
            "machine_101",
            "CRITICAL",
            "PWF",
        )

        self.assertIsNone(active_task)

    def test_status_is_normalized(self):
        update_task_status(self.task_id, " in_progress ")

        self.assertEqual(
            get_task(self.task_id)["status"], "IN_PROGRESS"
        )

    def test_invalid_status_is_rejected(self):
        with self.assertRaises(ValueError):
            update_task_status(self.task_id, "STARTED")

    def test_non_string_status_is_rejected(self):
        with self.assertRaises(ValueError):
            update_task_status(self.task_id, 123)

    def test_nonexistent_task_returns_false(self):
        updated = update_task_status(-999999, "IN_PROGRESS")

        self.assertFalse(updated)

    def test_nonexistent_task_returns_none(self):
        self.assertIsNone(get_task(-999999))

    def test_pending_can_move_to_in_progress(self):
        updated=update_task_status(
            self.task_id,"IN_PROGRESS"
        )

        self.assertTrue(updated)
        self.assertEqual(
            get_task(self.task_id)["status"],"IN_PROGRESS"
        )

    def test_pending_can_be_cancelled(self):
        updated=update_task_status(
            self.task_id,"CANCELLED"
        )

        self.assertTrue(updated)
        self.assertEqual(
            get_task(self.task_id)["status"], "CANCELLED"
        )

    def test_in_progress_can_be_completed(self):
        update_task_status(self.task_id, "IN_PROGRESS")

        updated = update_task_status(
            self.task_id, "COMPLETED"
        )

        self.assertTrue(updated)
        self.assertEqual(
            get_task(self.task_id)["status"], "COMPLETED"
        )

    def test_in_progress_can_be_cancelled(self):
        update_task_status(self.task_id, "IN_PROGRESS")

        updated = update_task_status(
            self.task_id, "CANCELLED"
        )

        self.assertTrue(updated)
        self.assertEqual(
            get_task(self.task_id)["status"], "CANCELLED"
        )

    def test_completed_task_cannot_be_reopened(self):
        update_task_status(self.task_id, "IN_PROGRESS")
        update_task_status(self.task_id, "COMPLETED")

        with self.assertRaises(ValueError):
            update_task_status(self.task_id, "IN_PROGRESS")

        self.assertEqual(
            get_task(self.task_id)["status"], "COMPLETED"
        )

    def test_cancelled_task_cannot_be_reopened(self):
        update_task_status(self.task_id, "CANCELLED")

        with self.assertRaises(ValueError):
            update_task_status(self.task_id, "PENDING")

        self.assertEqual(
            get_task(self.task_id)["status"], "CANCELLED"
        )

    def test_pending_cannot_skip_directly_to_completed(self):
        with self.assertRaises(ValueError):
            update_task_status(self.task_id, "COMPLETED")

        self.assertEqual(
            get_task(self.task_id)["status"], "PENDING"
        )

    def test_reapplying_same_status_is_allowed(self):
        updated = update_task_status(
            self.task_id, "PENDING"
        )

        self.assertTrue(updated)
        self.assertEqual(
            get_task(self.task_id)["status"], "PENDING"
        )



if __name__ == "__main__":
    unittest.main(verbosity=2)