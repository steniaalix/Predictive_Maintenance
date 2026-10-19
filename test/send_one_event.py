import json
import time
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt


BROKER="localhost"
PORT=1883
TOPIC="machine/sensors"

df=pd.read_csv("data/ai4i2020.csv")
failure_columns = ["TWF", "HDF", "PWF", "OSF"]
failure_rows=df[df[failure_columns].eq(1).any(axis=1)]

if failure_rows.empty:
    raise ValueError("No matching failure rows found in the dataset.")

row=failure_rows.iloc[0]

message={
    "machine_id":f"test_machine_{int(row["UDI"])}",
    "timestamp":datetime.now().isoformat(),
    "source":"one_shot_test",
    "type":row["Type"],
    "air_temperature": float(row["Air temperature [K]"]),
    "process_temperature": float(row["Process temperature [K]"]),
    "rotational_speed": float(row["Rotational speed [rpm]"]),
    "torque": float(row["Torque [Nm]"]),
    "tool_wear": float(row["Tool wear [min]"])
}

payload=json.dumps(message)

client=mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

try:
    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER,PORT,keepalive=60)
    client.loop_start()

    time.sleep(1)

    info=client.publish(TOPIC,payload,qos=1)

    info.wait_for_publish(timeout=10)

    if info.is_published():
        print("\nONE EVENT PUBLISHED SUCCESSFULLY")
        print("Topic:", TOPIC)
        print("Machine:", message["machine_id"])
        print("Sensor values:")
        print(json.dumps(message, indent=4))
    else:
        print("Publish was not confirmed.")
finally:
    client.loop_stop()
    client.disconnect()