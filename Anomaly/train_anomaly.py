import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)




df=pd.read_csv("data/ai4i2020.csv")

print("=" * 60)
print("ANOMALY DETECTION - TRAIN / VALIDATION / TEST")
print("=" * 60)

print("Total samples:", len(df))

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

y = df["Machine failure"]


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDATA SPLIT")

print("Training samples:   ", len(X_train))
print("Validation samples: ", len(X_val))
print("Test samples:       ", len(X_test))


model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

print("\nTraining Isolation Forest...")

model.fit(X_train)

print("Training completed.")


train_scores = model.decision_function(X_train)

val_scores = model.decision_function(X_val)

test_scores = model.decision_function(X_test)


print("\nSCORE STATISTICS")

print(
    "Training:",
    "min =", train_scores.min(),
    "max =", train_scores.max(),
    "mean =", train_scores.mean()
)

print(
    "Validation:",
    "min =", val_scores.min(),
    "max =", val_scores.max(),
    "mean =", val_scores.mean()
)

print(
    "Test:",
    "min =", test_scores.min(),
    "max =", test_scores.max(),
    "mean =", test_scores.mean()
)


percentiles = [
    1,
    2,
    5,
    10,
    15,
    20
]

candidate_thresholds = {}



print("\nCANDIDATE THRESHOLDS")

for percentile in percentiles:

    threshold = np.percentile(
        train_scores,
        percentile
    )

    candidate_thresholds[percentile] = threshold

    print(
        f"Lowest {percentile:>2}% "
        f"-> threshold = {threshold:.6f}"
    )



    print("\n")
print("=" * 60)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 60)

validation_results = []


for percentile, threshold in candidate_thresholds.items():

    # Lower score = anomaly
    val_predictions = (
        val_scores <= threshold
    ).astype(int)


    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    anomaly_count = val_predictions.sum()


    validation_results.append({

        "percentile": percentile,

        "threshold": threshold,

        "anomalies": anomaly_count,

        "precision": precision,

        "recall": recall,

        "f1": f1
    })


    print(
        f"\nLowest {percentile}%"
    )

    print(
        f"Threshold: {threshold:.6f}"
    )

    print(
        f"Anomalies: {anomaly_count}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )


best_result = max(
    validation_results,
    key=lambda x: x["f1"]
)

best_percentile = best_result["percentile"]

best_threshold = best_result["threshold"]


print("\n")
print("=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Selected percentile: "
    f"Lowest {best_percentile}%"
)

print(
    f"Selected threshold: "
    f"{best_threshold:.6f}"
)

print(
    f"Validation F1: "
    f"{best_result['f1']:.4f}"
)


print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_predictions = (
    test_scores <= best_threshold
).astype(int)


precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()


print("\nTEST RESULTS")

print(
    f"Threshold: {best_threshold:.6f}"
)

print(
    f"Anomalies detected: "
    f"{test_predictions.sum()}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)


print("\nCONFUSION MATRIX")

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\n")
print("=" * 60)
print("ANOMALY DETECTION EXPERIMENT SUMMARY")
print("=" * 60)

print(
    "Model: Isolation Forest"
)

print(
    "Trees:",
    200
)

print(
    "Contamination:",
    "auto"
)

print(
    "Selected threshold:",
    round(best_threshold, 6)
)

print(
    "Validation F1:",
    round(best_result["f1"], 4)
)

print(
    "Test Precision:",
    round(precision, 4)
)


print(
    "Test Recall:",
    round(recall, 4)
)

print(
    "Test F1:",
    round(f1, 4)
)

print("\nExperiment completed.")