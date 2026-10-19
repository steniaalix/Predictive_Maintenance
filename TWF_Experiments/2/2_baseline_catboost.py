from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
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



df=pd.read_csv("data/ai4i2020.csv")
df.drop(columns=["UDI","Product ID"],inplace=True)

df["Power"]=df["Torque [Nm]"]*df["Rotational speed [rpm]"]


type_mapping={
    "L":0,
    "M":1,
    "H":2
}


df["Type"]=df["Type"].map(type_mapping)


target="TWF"

failure_columns=["Machine failure","HDF","TWF","PWF","OSF","RNF"]

X=df.drop(columns=failure_columns)

y=df[target]


X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


model=CatBoostClassifier(
    loss_function="Logloss",
    random_seed=42,
    verbose=False,
    thread_count=-1,
    class_weights=(1,10)
)



param_grid={
    "iterations":[100,200,300],
    "depth":[4,6,8],
    "learning_rate":[0.03,0.05,0.1],
    "l2_leaf_reg":[1,3,5]
}


grid=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train,y_train)


print("\n" + "=" * 60)
print("CATBOOST — GRID SEARCH RESULTS")
print("=" * 60)

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV F1:")
print(f"{grid.best_score_:.4f}")

best_model=grid.best_estimator_

cv_probs=cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:,1]


print("\n" + "=" * 60)
print("OOF PROBABILITY INFORMATION")
print("=" * 60)

print(f"Minimum probability: {cv_probs.min()}")
print(f"Maximum probability: {cv_probs.max()}")
print(f"Mean probability   : {cv_probs.mean()}")


thresholds=np.arange(0.01,0.51,0.01)

best_threshold=0
best_f1=0

print("\n" + "=" * 60)
print("OOF THRESHOLD SEARCH")
print("=" * 60)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<8}"
    f"{'FN':<8}"
)



for threshold in thresholds:
    predictions=(cv_probs>=threshold).astype(int)

    precision=precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall=recall_score(
        y_train,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<8}"
        f"{fn:<8}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

threshold=best_threshold

print("\nSelected threshold:", threshold)
print("Best OOF F1:", best_f1)

test_probs=best_model.predict_proba(
    X_test
)[:,1]

test_predictions=(test_probs>=threshold).astype(int)


tn,fp,fn,tp=confusion_matrix(
    y_test,
    test_predictions,
    labels=[0,1]
).ravel()

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
    test_probs
)

pr_auc = average_precision_score(
    y_test,
    test_probs
)


print("\n" + "=" * 60)
print("CATBOOST — TWF — FINAL TEST RESULTS")
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