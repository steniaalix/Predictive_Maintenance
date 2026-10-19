from notification_agent import process_machine_event
from database.notification_state import reset_state
prediction_result={
    "TWF":{
        "probability":0.94,
        "threshold":0.91,
        "prediction":1
    },
    "HDF": {
        "probability": 0.10,
        "threshold": 0.91,
        "prediction": 0
    },
    "PWF": {
        "probability": 0.02,
        "threshold": 0.03,
        "prediction": 0
    },
    "OSF": {
        "probability": 0.20,
        "threshold": 0.78,
        "prediction": 0
    },
    "RNF": {
        "supported": False,
        "message": "RNF prediction is not reliable."
    }

}

anomaly_result={
    "anomaly":True,
    "score":-0.08,
    "threshold":-0.065427
}

machine_id="test_limit_machine_4"
reset_state(machine_id)
for i in range(1,5):
    print("\n"+"="*60)
    print(f"NOTIFICATION TEST #{i}")
    print("=" * 60)

    result=process_machine_event(
        machine_id=machine_id,
        prediction_result=prediction_result,
        anomaly_result=anomaly_result
    )

    print("Severity:", result["severity"])
    print(
        "Notification sent:",
        result["notification_sent"]
    )

    print(
        "Message:",
        result.get("message", "Notification sent.")
    )

    if "notification_count" in result:
        print("Notification count:",result["notification_count"])