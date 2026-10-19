from notification_agent import process_machine_event
from dotenv import load_dotenv
from database.notification_state import reset_state

load_dotenv()
machine_id="LANGSMITH_TEST_001"
reset_state(machine_id)

prediction_result = {
    "TWF": {
        "prediction": 1,
        "probability": 0.94,
        "threshold": 0.91
    },
    "HDF": {
        "prediction": 0,
        "probability": 0.10,
        "threshold": 0.91
    },
    "PWF": {
        "prediction": 0,
        "probability": 0.01,
        "threshold": 0.03
    },
    "OSF": {
        "prediction": 0,
        "probability": 0.12,
        "threshold": 0.78
    }
}

anomaly_result={
    "anomaly":True,
    "score":-0.08,
    "threshold":-0.065427
}

print("Running one notification-agent test...")

result = process_machine_event(
    machine_id=machine_id,
    prediction_result=prediction_result,
    anomaly_result=anomaly_result
)

print("\nReturned result:")
print(result)