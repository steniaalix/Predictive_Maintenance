import json
import paho.mqtt.client as mqtt

from prediction import predict_failures
from anomaly_detection import detect_anomaly
from notification_agent import process_machine_event

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

        notification_agent=process_machine_event(
            machine_id=data['machine_id'],
            prediction_result=prediction_result,
            anomaly_result=anomaly_result
        )


        print("\n" + "-" * 60)

        print("NOTIFICATION AGENT")

        print("-" * 60)

        print("severity:",notification_agent['severity'])

        print("Notification sent:",notification_agent['notification_sent'])

        




    except Exception as e:

        print("\nError processing message:")
        print(e)

client=mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect=on_connect
client.on_message=on_message


print("=" * 60)
print("MQTT PREDICTION + ANOMALY CONSUMER")
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