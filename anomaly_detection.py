import joblib
import pandas as pd

artifact = joblib.load(
    "app/anomaly_model.joblib"
)

model = artifact["model"]

threshold = artifact["threshold"]

feature_order = artifact["feature_order"]

def detect_anomaly(
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):

    power = (
        torque
        * rotational_speed
    )

    input_data = pd.DataFrame([{

        "Air temperature [K]": air_temperature,

        "Process temperature [K]": process_temperature,

        "Rotational speed [rpm]": rotational_speed,

        "Torque [Nm]": torque,

        "Tool wear [min]": tool_wear,

        "Power": power

    }])

    input_data = input_data[
        feature_order
    ]


    score = model.decision_function(
        input_data
    )[0]

    anomaly = (
        score <= threshold
    )

    return {

        "anomaly": bool(anomaly),

        "score": round(
            float(score),
            6
        ),

        "threshold": threshold

    }