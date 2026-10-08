import sqlite3
from datetime import datetime

DB_PATH="data/notification_state.db"

def create_database():
    connection=sqlite3.connect(DB_PATH)

    cursor=connection.cursor()

    cursor.execute("""

CREATE TABLE IF NOT EXISTS notification_state(
machine_id TEXT PRIMARY KEY,
severity TEXT,
failure_type TEXT,
notification_count INTEGER DEFAULT 0,
last_notification_time TEXT
)
""")

    connection.commit()
    connection.close()

def get_state(machine_id):
    connection=sqlite3.connect(DB_PATH)
    cursor=connection.cursor()

    cursor.execute("""

SELECT machine_id,severity,failure_type,notification_count,last_notification_time
FROM notification_state WHERE machine_id=?""",(machine_id,))

    row=cursor.fetchone()
    connection.close()

    if row is None:
        return None

    return {
        "machine_id":row[0],
        "severity":row[1],
        "failure_type":row[2],
        "notification_count":row[3],
        "last_notification_time":row[4]
    }

def save_state(machine_id,severity,failure_type,notification_count):
    connection=sqlite3.connect(DB_PATH)

    cursor=connection.cursor()

    current_time=datetime.now().isoformat()

    cursor.execute("""

INSERT OR REPLACE INTO notification_state(
    machine_id,
    severity,
    failure_type,
    notification_count,
    last_notification_time
    )VALUES(?,?,?,?,?)"""
    ,(machine_id,severity,failure_type,notification_count,current_time
    ))

    connection.commit()
    connection.close()

def reset_state(machine_id):
    connection=sqlite3.connect(DB_PATH)

    cursor=connection.cursor()

    cursor.execute("""

DELETE FROM notification_state
WHERE machine_id=?
""",
(machine_id,))
    connection.commit()
    connection.close()