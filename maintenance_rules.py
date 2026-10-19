FAILURE_RULES = {
    "TWF": {"assigned_role": "Tooling Maintenance Team", "recommended_action": "Inspect tool wear and assess whether the tool needs replacement."},
    "HDF": {"assigned_role": "Thermal and Mechanical Maintenance Team", "recommended_action": "Inspect the machine's thermal operating conditions and cooling system."},
    "PWF": {"assigned_role": "Equipment Maintenance Team", "recommended_action": "Inspect power-related operating conditions and verify the sensor readings."},
    "OSF": {"assigned_role": "Mechanical Maintenance Team", "recommended_action": "Inspect the machine's mechanical operating conditions and tool loading."},
}
SEVERITY_RULES = {
    "CRITICAL": {"priority": "HIGH", "response_target": "Inspect promptly"},
    "WARNING": {"priority": "MEDIUM", "response_target": "Review during the next maintenance assessment"},
}
def assign_maintenance(severity, failure_type=None, anomaly=False):
    severity = severity.strip().upper() if isinstance(severity, str) else ""
    failure_type = failure_type.strip().upper() if isinstance(failure_type, str) and failure_type.strip() else None
    if severity == "NORMAL":
        return None
    if severity not in SEVERITY_RULES:
        raise ValueError(f"Unsupported severity: {severity}")
    severity_rule = SEVERITY_RULES[severity]
    if failure_type in FAILURE_RULES:
        failure_rule = FAILURE_RULES[failure_type]
        assigned_role = failure_rule["assigned_role"]
        recommended_action = failure_rule["recommended_action"]
    elif severity == "WARNING" and anomaly:
        assigned_role = "Equipment Inspection Team"
        recommended_action = "Review the unusual operating readings and verify the sensor measurements."
    elif severity == "CRITICAL":
        assigned_role = "General Maintenance Team"
        recommended_action = "Inspect the machine and verify the reported condition before deciding on corrective work."
    else:
        raise ValueError("A WARNING requires a known failure type or anomaly=True.")
    return {"assigned_role": assigned_role, "priority": severity_rule["priority"],
            "response_target": severity_rule["response_target"], "recommended_action": recommended_action}
