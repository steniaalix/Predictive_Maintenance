import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)

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


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")


# ============================================================
# 2. REMOVE IDENTIFIERS
# ============================================================

df = df.drop(columns=["UDI", "Product ID"])


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["Power"] = (
    df["Torque [Nm]"] *
    df["Rotational speed [rpm]"]
)


# ============================================================
# 4. ENCODE TYPE
# ============================================================

type_mapping = {
    "L": 0,
    "M": 1,
    "H": 2
}

df["Type"] = df["Type"].map(type_mapping)


# ============================================================
# 5. TARGET
# ============================================================

target = "TWF"

failure_columns = [
    "Machine failure",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "TWF"
]

X = df.drop(columns=failure_columns)
y = df[target]

X.columns = (
    X.columns
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace("<", "", regex=False)
)
# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("Training class distribution:")
print("Class 0:", negative_count)
print("Class 1:", positive_count)
print("Class ratio:", scale_pos_weight)


# ============================================================
# 8. STRATIFIED CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. LIGHTGBM MODEL
# ============================================================

lgbm = LGBMClassifier(
    objective="binary",
    random_state=42,
    n_jobs=1,
    verbosity=-1,
    scale_pos_weight=scale_pos_weight
)


# ============================================================
# 10. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [-1, 3, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}


# ============================================================
# 11. GRID SEARCH
# ============================================================

grid = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)


# ============================================================
# 12. BEST MODEL
# ============================================================

best_model = grid.best_estimator_

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV F1:")
print(grid.best_score_)


# ============================================================
# 13. OOF PREDICTIONS
# ============================================================

oof_probabilities = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


# ============================================================
# 14. THRESHOLD SELECTION
# ============================================================

best_threshold = 0.5
best_oof_f1 = 0

for threshold in np.arange(0.01, 0.51, 0.01):

    oof_predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    if score > best_oof_f1:
        best_oof_f1 = score
        best_threshold = threshold


print("\nSelected Threshold:", best_threshold)
print("Best OOF F1:", best_oof_f1)


# ============================================================
# 15. FINAL TEST PREDICTION
# ============================================================

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions
).ravel()


# ============================================================
# 17. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

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

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


# ============================================================
# 18. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LIGHTGBM — TWF — STANDARDIZED FINAL RESULTS")
print("=" * 60)

print(f"Threshold: {best_threshold:.2f}")

print("\nConfusion matrix:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nTest metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print("=" * 60)