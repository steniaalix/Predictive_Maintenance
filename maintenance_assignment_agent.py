from maintenance_rules import assign_maintenance
from maintenance_task_writer import generate_task_description
from database.maintenance_database import create_task, get_task, get_active_task_for_incident

def create_maintenance_assignment(machine_id, severity, failure_type=None, failure_probability=None, anomaly=False):
    if not isinstance(machine_id, str) or not machine_id.strip():
        raise ValueError("machine_id must be a non-empty string.")
    machine_id = machine_id.strip()
    if not isinstance(severity, str):
        raise ValueError("severity must be a string.")
    severity = severity.strip().upper()
    if isinstance(failure_type, str):
        failure_type = failure_type.strip().upper() or None
    if failure_probability is not None:
        if not isinstance(failure_probability, (int, float)):
            raise ValueError("failure_probability must be a number or None.")
        if not 0 <= failure_probability <= 1:
            raise ValueError("failure_probability must be between 0 and 1.")

    assignment = assign_maintenance(severity=severity, failure_type=failure_type, anomaly=anomaly)
    if assignment is None:
        return {"task_created": False, "duplicate": False, "task_id": None, "task": None,
                "message": "No maintenance task required for NORMAL severity."}

    existing_task = get_active_task_for_incident(machine_id, severity, failure_type)
    if existing_task is not None:
        return {"task_created": False, "duplicate": True, "task_id": existing_task["task_id"],
                "task": existing_task, "message": "An active task already exists for this incident."}

    description = generate_task_description(machine_id, severity, failure_type, failure_probability, assignment)
    task_id = create_task(machine_id, severity, failure_type, failure_probability,
                        assignment["assigned_role"], assignment["priority"],
                        assignment["response_target"], description)
    saved_task = get_task(task_id)
    if saved_task is None:
        raise RuntimeError(f"Task {task_id} was created but could not be retrieved.")
    return {"task_created": True, "duplicate": False, "task_id": task_id, "task": saved_task,
            "message": "Maintenance task created successfully."}
