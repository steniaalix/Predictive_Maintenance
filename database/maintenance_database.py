import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "maintenance_tasks.db"
ACTIVE_STATUSES = ("PENDING", "IN_PROGRESS")
@contextmanager
def database_connection():
    connection=sqlite3.connect(DATABASE_PATH)

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

def create_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with database_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                failure_type TEXT,
                failure_probability REAL,
                assigned_role TEXT NOT NULL,
                priority TEXT NOT NULL,
                response_target TEXT NOT NULL,
                task_description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

def create_task(machine_id, severity, failure_type, failure_probability, assigned_role, priority, response_target, task_description):
    create_database()
    current_time = datetime.now(timezone.utc).isoformat()
    with database_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO maintenance_tasks (
                machine_id, severity, failure_type, failure_probability,
                assigned_role, priority, response_target, task_description,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (machine_id, severity, failure_type, failure_probability, assigned_role,
            priority, response_target, task_description, "PENDING", current_time, current_time))
        task_id = cursor.lastrowid
    return task_id

def get_task(task_id):
    create_database()
    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM maintenance_tasks WHERE task_id = ?", (task_id,)).fetchone()
    return None if row is None else dict(row)

def get_all_tasks():
    create_database()
    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM maintenance_tasks ORDER BY created_at DESC, task_id DESC").fetchall()
    return [dict(row) for row in rows]

def get_active_task_for_incident(machine_id, severity, failure_type):
    create_database()
    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("""
            SELECT * FROM maintenance_tasks
            WHERE machine_id = ? AND severity = ? AND failure_type IS ?
            AND status IN (?, ?)
            ORDER BY task_id DESC LIMIT 1
        """, (machine_id, severity, failure_type, *ACTIVE_STATUSES)).fetchone()
    return None if row is None else dict(row)


def update_task_status(task_id, new_status):
    allowed_statuses={
        "PENDING",
        "IN_PROGRESS",
        "COMPLETED",
        "CANCELLED",
    }

    if not isinstance(new_status,str):
        raise ValueError("Task status must be a string.")

    new_status=new_status.strip().upper()

    if new_status not in allowed_statuses:
        raise ValueError(f"Invalid task status: {new_status}")
    allowed_transitions={
        "PENDING":{"IN_PROGRESS","CANCELLED"},
        "IN_PROGRESS":{"COMPLETED","CANCELLED"},
        "COMPLETED":set(),
        "CANCELLED":set(),
    }

    create_database()

    current_time=datetime.now(timezone.utc).isoformat()

    with  database_connection() as connection:
        row=connection.execute(
            """
            SELECT status
            FROM maintenance_tasks
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()

        if row is None:
            return False
        current_status=row[0]

        if current_status==new_status:
            return True

        if new_status not in allowed_transitions[current_status]:
            raise ValueError(
                f"Invalid task status transition:"
                f"{current_status}->{new_status}"
            )

        cursor=connection.execute(
            """
            UPDATE maintenance_tasks
            SET status = ?, updated_at = ?
            WHERE task_id = ? AND status = ?
            """,
            (new_status,current_time,task_id,current_status),
        )

        if cursor.rowcount==0:
            raise RuntimeError(
                "Task status changed concurrently. Please retry."
            )

    return True