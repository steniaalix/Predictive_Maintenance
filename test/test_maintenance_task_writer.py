import unittest
from maintenance_rules import assign_maintenance
from maintenance_task_writer import generate_task_description

class TestMaintenanceTaskWriter(unittest.TestCase):
    def test_generates_non_empty_description(self):
        assignment = assign_maintenance("CRITICAL", "PWF")
        description = generate_task_description("MAINTENANCE_WRITER_TEST", "CRITICAL", "PWF", 0.94, assignment)
        self.assertIsInstance(description, str)
        self.assertTrue(description.strip())

if __name__ == "__main__":
    unittest.main()
