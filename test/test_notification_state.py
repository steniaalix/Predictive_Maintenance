from database.notification_state import (
    create_database,get_state,save_state,reset_state
)


create_database()

machine_id="test_machine_001"

print("\nSaving state...")

save_state(
    machine_id=machine_id,
    severity="CRITICAL",
    failure_type="TWF",
    notification_count=3
)


print("\nReading state...")

state=get_state(machine_id)

print(state)

print("\nResetting state...")

reset_state(machine_id)

print("\nReading after reset...")

state = get_state(machine_id)

print(state)