import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/ai4i2020.csv")



df["Power"] = (
    df["Torque [Nm]"]
    * df["Rotational speed [rpm]"]
)

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power"
]

X = df[features]

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

print("Training final anomaly detection model...")

model.fit(X)

print("Training completed.")

threshold = -0.065427


artifact = {

    "model": model,

    "threshold": threshold,

    "feature_order": features,

    "n_estimators": 200,

    "contamination": "auto",

    "anomaly_features": features,

    "validation_metrics": {
        "threshold_percentile": 5,
        "validation_f1": 0.1926
    },

    "test_metrics": {
        "precision": 0.2258,
        "recall": 0.2745,
        "f1": 0.2478
    },

    "sklearn_version": __import__(
        "sklearn"
    ).__version__
}

output_path = "app/anomaly_model.joblib"

joblib.dump(
    artifact,
    output_path
)


print("\n" + "=" * 60)
print("ANOMALY MODEL SAVED")
print("=" * 60)

print("Path:", output_path)
print("Threshold:", threshold)
print("Features:", len(features))
print("Trees:", 200)

print("\nArtifact contents:")

for key in artifact:
    print("-", key)