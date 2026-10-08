import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


df=pd.read_csv("data/ai4i2020.csv")

df.drop(columns=["UDI","Product ID"],inplace=True)

df["Power"]=df["Torque [Nm]"]*df["Rotational speed [rpm]"]

type_mapping={
    "L":0,
    "M":1,
    "H":2
}

df["Type"]=df["Type"].map(type_mapping)

target="RNF"

failure_columns=[
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

X=df.drop(columns=failure_columns)

y=df[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

negative_count=(y_train==0).sum()
positive_count=(y_train==1).sum()

class_ratio=negative_count/positive_count

print("Training class distribution:")
print("Class 0:", negative_count)
print("Class 1:", positive_count)
print("Class ratio:", class_ratio)


cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


rf=RandomForestClassifier(class_weight="balanced",
                        random_state=42,
                        n_jobs=-1)


param_grid={
    "n_estimators":[200,300,500],
    "max_depth":[None,5,10],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2],
    "max_features":["sqrt","log2"]
}

grid=GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train,y_train)

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV F1:")
print(grid.best_score_)

best_model=grid.best_estimator_


oof_probabilities=cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:,1]


best_threshold=0.5
best_f1=0

for threshold in np.arange(0.01,1.0,00.01):
    oof_predictions=(oof_probabilities>=threshold).astype(int)
    f1=f1_score(y_train,oof_predictions,zero_division=0)

    if f1>best_f1:
        best_f1=f1
        best_threshold=threshold

print("\nSelected Threshold:",best_threshold)
print("\nBest OOF F1:",best_f1)

test_probabilities=best_model.predict_proba(X_test)[:,1]

test_predictions=(test_probabilities>=best_threshold).astype(int)

tn,fp,fn,tp=confusion_matrix(y_test,test_predictions).ravel()



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
# 18. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST — RNF — STANDARDIZED FINAL RESULTS")
print("=" * 60)

print(f"Threshold: {best_threshold:.2f}")

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