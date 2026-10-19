import unittest
from maintenance_rules import assign_maintenance

class TestMaintenanceRules(unittest.TestCase):
    def test_critical_pwf(self):
        result = assign_maintenance("CRITICAL", "PWF")
        self.assertEqual(result["assigned_role"], "Equipment Maintenance Team")
        self.assertEqual(result["priority"], "HIGH")
        self.assertEqual(result["response_target"], "Inspect promptly")
    def test_critical_twf(self):
        self.assertEqual(assign_maintenance("CRITICAL", "TWF")["assigned_role"], "Tooling Maintenance Team")
    def test_warning_anomaly(self):
        result = assign_maintenance("WARNING", anomaly=True)
        self.assertEqual(result["assigned_role"], "Equipment Inspection Team")
        self.assertEqual(result["priority"], "MEDIUM")
    def test_normal_returns_none(self):
        self.assertIsNone(assign_maintenance("NORMAL"))
    def test_unknown_severity_raises(self):
        with self.assertRaises(ValueError):
            assign_maintenance("UNKNOWN")
    def test_warning_without_type_or_anomaly_raises(self):
        with self.assertRaises(ValueError):
            assign_maintenance("WARNING")

if __name__ == "__main__":
    unittest.main()
