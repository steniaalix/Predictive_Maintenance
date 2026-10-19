import json
import paho.mqtt.client as mqtt
from maintenance_assignment_agent import create_maintenance_assignment
from prediction import predict_failures
from anomaly_detection import detect_anomaly
from notification_agent import process_machine_event
import traceback

BROKER="localhost"
PORT=1883
TOPIC="machine/sensors"
def on_connect(client,userdata,flags,reason_code,properties):
    print("\nConnected to MQTT broker")

    client.subscribe(TOPIC)
    print("Subscribe to:",TOPIC)
    print("Watiting for sensor data...\n")


def on_message(client,userdata,msg):

    print("\n"+"="*60)
    print("MESSAGE RECEIVED")
    print("="*60)
    try:
        data=json.loads(msg.payload.decode())


        print("Machine ID:",data["machine_id"])
        print("Timestamp:",data["timestamp"])

        prediction_result=predict_failures(
            machine_type=data["type"],
            air_temperature=data['air_temperature'],
            process_temperature=data["process_temperature"],
            rotational_speed=data["rotational_speed"],
            torque=data["torque"],
            tool_wear=data["tool_wear"]
        )

        anomaly_result=detect_anomaly(
            air_temperature=data['air_temperature'],
            process_temperature=data['process_temperature'],
            rotational_speed=data['rotational_speed'],
            torque=data['torque'],
            tool_wear=data['tool_wear']
        )

        print("\n" + "-" * 60)
        print("\nFAILURE RESULTS")
        print("\n" + "-" * 60)
        print(prediction_result)

        print("\n" + "-" * 60)
        print("\nANOMALY DETECTION")
        print("\n" + "-" * 60)
        print(anomaly_result)
        machine_id=data['machine_id']
        notification_agent=process_machine_event(
            machine_id=machine_id,
            prediction_result=prediction_result,
            anomaly_result=anomaly_result
        )

        print("\nFULL NOTIFICATION RESULT")
        print(notification_agent)
        severity=notification_agent.get(
            "severity","NORMAL"
        )

        severity=(
            severity.upper()
            if isinstance(severity,str)
            else "NORMAL"
        )

        if severity in {"WARNING","CRITICAL"}:
            failure_type,failure_probability=(
                get_primary_failure(prediction_result)
            )

            anomaly=bool(anomaly_result.get("anomaly",False))

            try:
                maintenance_result=create_maintenance_assignment(
                    machine_id=machine_id,
                    severity=severity,
                    failure_type=failure_type,
                    failure_probability=failure_probability,
                    anomaly=anomaly
                )
                if maintenance_result.get("task_created"):
                    task=maintenance_result["task"]

                    print("\n" + "=" * 50)
                    print("MAINTENANCE TASK CREATED")
                    print("=" * 50)

                    print("Task ID:", task["task_id"])
                    print("Machine:", task["machine_id"])
                    print("Severity:", task["severity"])
                    print("Failure type:", task["failure_type"])
                    print("Assigned team:", task["assigned_role"])
                    print("Priority:", task["priority"])
                    print("Response target:", task["response_target"])
                    print("Status:", task["status"])
                    print("Description:", task["task_description"])
                elif maintenance_result.get("duplicate"):
                    print("Ann active maintenance task already exists:",
                        maintenance_result.get("task_id"))

                else:
                    print(maintenance_result.get("message","No maintenance task created."))
            except Exception as exc:
                print("Maintenance assignment failed:",repr(exc))

        print("\n" + "-" * 60)

        print("NOTIFICATION AGENT")

        print("-" * 60)

        print("severity:",notification_agent['severity'])

        print("Notification sent:",notification_agent['notification_sent'])

        




    except Exception as e:

        print("\nError processing message:")
        traceback.print_exc()




def get_primary_failure(prediction_result):
    predicted_failures=[]

    for failure_type,details in prediction_result.items():
        if not isinstance(details,dict):
            continue

        if details.get("prediction",0)==1:
            probability=details.get("probability")
            if probability is not None:
                probability=float(probability)
            predicted_failures.append(
                (failure_type,probability)
            )

    if not predicted_failures:
        return None,None

    predicted_failures.sort(key=lambda item: (
        item[1] is not None,
        item[1] if item[1] is not None else -1
    ),reverse=True
    )

    return predicted_failures[0]





client=mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect=on_connect
client.on_message=on_message





def main():
    print("="*60)
    print("MQTT PREDICTION + ANOMALY CONSUMER")
    print("="*60)

    print("Broker:",BROKER)
    print("Port:",PORT)
    print("Topic:",TOPIC)

    client.connect(BROKER,PORT,60)

    client.loop_forever()

if __name__=="__main__":
    main()


print("=" * 60)

print("Broker:", BROKER)
print("Port:", PORT)
print("Topic:", TOPIC)


client.connect(
    BROKER,
    PORT,
    60
)
client.loop_forever()