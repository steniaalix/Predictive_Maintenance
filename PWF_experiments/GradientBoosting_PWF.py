import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
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



target="PWF"

failure_columns=["Machine failure","TWF","HDF","PWF","OSF","RNF"]


X=df.drop(columns=failure_columns)

y=df[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


negative_count=(y_train==0).sum()
positive_count=(y_train==1).sum()

class_ratio=negative_count/positive_count

sample_weights=np.where(
    y_train==1,
    class_ratio,
    1.0
)


print("Training samples:",len(X_train))
print("Positive samples:",positive_count)
print("Neggative samples:",negative_count)
print("Class ratio:",class_ratio)

model=GradientBoostingClassifier(random_state=42)

param_grid={
    "n_estimators":[100,200],
    "learning_rate":[0.03,0.05,0.1],
    "max_depth":[2,3],
    "min_samples_leaf":[1,2],
    "subsample":[0.8,1.0]
}


grid=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train,y_train,sample_weight=sample_weights)

print("\nBEST PARAMETERS:")
print(grid.best_params_)

print("\nBEST CV F1:")
print(grid.best_score_)

best_model=GradientBoostingClassifier(
    **grid.best_params_,
    random_state=42
)

oof_probs=cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
    params={
        "sample_weight":sample_weights
    }
)[:,1]


thresholds=np.arange(0.01,1.0,0.01)

best_threshold=0.5
best_oof_f1=-1

for threshold in thresholds:
    oof_predictions=(oof_probs>=threshold).astype(int)

    score=f1_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    if score>best_oof_f1:
        best_oof_f1=score
        best_threshold=threshold

print("\nSELECTED THRESHOLD:")
print(best_threshold)

print("\nBEST OOF F1:")
print(best_oof_f1)


final_model=GradientBoostingClassifier(
    **grid.best_params_,
    random_state=42
)


final_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


test_probs = final_model.predict_proba(
    X_test
)[:, 1]


test_predictions = (
    test_probs >= best_threshold
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
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


print("\n" + "=" * 50)
print("GRADIENTBOOSTING-PWF-STANDARDIZED FINAL RESULT")
print("=" * 50)
print("Threshold:", best_threshold)

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("ROC-AUC  :", roc_auc)
print("PR-AUC   :", pr_auc)
print("\n" + "=" * 50)

artifact = {
    "failure_type": target,
    "model": final_model,
    "threshold": best_threshold,
    "feature_order": list(X.columns),
    "type_mapping": type_mapping,
    "best_parameters": grid.best_params_,

    "metrics": {
        "best_cv_f1": grid.best_score_,
        "oof_f1": best_oof_f1,

        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    },

    "sklearn_version": __import__("sklearn").__version__
}

joblib.dump(
    artifact,
    "app/pwf_final_model.joblib"
)