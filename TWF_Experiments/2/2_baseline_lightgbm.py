import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,GridSearchCV,cross_val_predict)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from lightgbm import LGBMClassifier


df=pd.read_csv("data/ai4i2020.csv")

df=df.drop(columns=["UDI","Product ID"])

df["Power"]=(
    df["Torque [Nm]"]*
    df["Rotational speed [rpm]"]
)


type_mapping={
    "L":0,
    "M":1,
    "H":2
}

df["Type"]=df["Type"].map(type_mapping)

target="TWF"

failure_columns=[
    "Machine failure",
    "HDF",
    "PWF",
    "TWF",
    "OSF",
    "RNF"
]

X=df.drop(columns=failure_columns)
y=df[target]

X.columns = (
    X.columns
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace("<", "", regex=False)
)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.3,
    stratify=y
)


model=LGBMClassifier(
    objective="binary",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


param_grid={
    "n_estimators":[100,200,300],
    "max_depth":[-1,3,5],
    "learning_rate":[0.03,0.05,0.1],
    "num_leaves":[15,31,63],
    "subsample":[0.8,1.0],
    "colsample_bytree":[0.8,1.0]
}

grid=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train,y_train)

best_model=grid.best_estimator_

cv_probs=cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=5,
    method="predict_proba",
    n_jobs=-1
)[:,1]

print("Minimum probability:",cv_probs.min())
print("Maximum probability:",cv_probs.max())
print("Mean probability:",cv_probs.mean())


thresholds=np.arange(0.01,1.0,0.01)

best_threshold=0
best_f1=0

for threshold in thresholds:
    preds=(cv_probs>=threshold).astype(int)

    precision = precision_score(
        y_train,
        preds,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        preds,
        zero_division=0
    )

    tn,fp,fn,tp=confusion_matrix(
        y_train,
        preds,
        labels=[0,1]
    ).ravel()

    print(
        f"{threshold:.2f}"
        f"{precision:.4f}"
        f"{recall:.4f}"
        f"{f1:.4f}"
        f"{fp}"
        f"{fn}"
    )

    if f1>best_f1:
        best_f1=f1
        best_threshold=threshold

print("\nSelected threshold:",best_threshold)
print("Best OOF F1:",best_f1)
best_model = grid.best_estimator_

test_probs = best_model.predict_proba(X_test)[:, 1]

threshold = 0.04

test_preds = (
    test_probs >= threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_preds,
    labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds, zero_division=0)
recall = recall_score(y_test, test_preds, zero_division=0)
f1 = f1_score(y_test, test_preds, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)

print("=" * 60)
print("LIGHTGBM — TWF — OOF SELECTED THRESHOLD")
print("=" * 60)

print(f"Threshold: {threshold:.2f}")

print("\nConfusion matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nTest metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print("=" * 60)