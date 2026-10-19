import json 
import time
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt


BROKER="localhost"
PORT=1883
TOPIC="machine/sensors"

DATASET="data/ai4i2020.csv"

INTERVAL=3

df=pd.read_csv(DATASET)

print("Dataset loaded")
print("Rows:",len(df))

client=mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.connect(
    BROKER,
    PORT,
    60
)

client.loop_start()

print(f"Connected to MQTT broker: {BROKER}:{PORT}")
print(f"Publishing to: {TOPIC}")
print(f"Interval: {INTERVAL} seconds")

print("\nStarting historical replay...\n")

try:
    for index,row in df.iterrows():
        message={
            "machine_id":f"machine_{int(row['UDI'])}",
            "timestamp":datetime.now().isoformat(),
            "source":"AI4I_2020_historical_replay",
            "type":row["Type"],
            "air_temperature":float(
                row["Air temperature [K]"]
            ),
            "process_temperature":float(
                row["Process temperature [K]"]
            ),
            "rotational_speed":float(
                row["Rotational speed [rpm]"]
            ),
            "torque":float(
                row["Torque [Nm]"]
            ),
            "tool_wear":float(
                row["Tool wear [min]"]
            )
        }

        payload=json.dumps(message)
        client.publish(
            TOPIC,
            payload
        )

        print(f"[{index+1}/{len(df)}]"
            f"{payload}"
        )

        time.sleep(INTERVAL)
except KeyboardInterrupt:
    print("\nReplay stopped by user.")


finally:
    client.loop_stop()
    client.disconnect()
    print("disconnected from MQTT broker.")