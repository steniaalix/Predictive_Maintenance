import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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

target="PWF"

failure_cloumns=["Machine failure","TWF","HDF","PWF","OSF","RNF"]

X=df.drop(columns=failure_cloumns)
y=df[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


negative_count=(y_train==0).sum()
positive_count=(y_train==1).sum()

class_ratio= negative_count/positive_count

print("=" * 60)
print("MLP NEURAL NETWORK - TWF")
print("=" * 60)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("Negative:", negative_count)
print("Positive:", positive_count)

print(
    "Class ratio:",
    class_ratio
)

sample_weights=np.where(
    y_train==1,
    class_ratio,
    1.0
)

cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


pipeline=Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "mlp",
        MLPClassifier(
            random_state=42,
            max_iter=500
        )
    )
])


param_grid={
    "mlp__hidden_layer_sizes":[
        (32,),
        (64,),
        (64,32)
    ],
    "mlp__activation":[
        "relu"
    ],
    "mlp__alpha":[
        0.0001,
        0.001
    ],
    "mlp__learning_rate_init":[
        0.001,
        0.01
    ]
}



grid=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(
    X_train,
    y_train,
    mlp__sample_weight=sample_weights
)

print("\nBEST PARAMETERS:")

print("\nBEST CV F1:",grid.best_score_)


best_model=grid.best_estimator_

oof_probs=cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
    params={
        "mlp__sample_weight":sample_weights
    }
)[:,1]


thresholds=np.arange(0.01,1.0,0.01)

best_threshold=0.5

best_f1=-1


for threshold in thresholds:
    oof_predictions=(
        oof_probs>=threshold
    ).astype(int)

    f1=f1_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    if f1>best_f1:
        best_f1=f1
        best_threshold=threshold



print(
    f"\nSELECTED THRESHOLD: "
    f"{best_threshold:.2f}"
)

print(
    "BEST OOF F1:",
    best_f1
)



# final_model=Pipeline([
#     (
#         "scaler",
#         StandardScaler()
#     ),
#     (
#         "mlp",
#         MLPClassifier(
#             **{
#                 key.replace("mlp__",""):value
#                 for key, value in grid.best_params_.items()
#             },
#             random_state=42,
#             max_iter=500
#         )
#     )
# ]
# )

final_model=grid.best_estimator_


final_model.fit(
    X_train,
    y_train,
    mlp__sample_weight=sample_weights
)


test_probs=final_model.predict_proba(
    X_test
)[:,1]

test_predictions=(
    test_probs>=best_threshold
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




print("\n")
print("=" * 60)
print("MLP-PWF-FINAL RESULT")
print("=" * 60)
print(
    f"Threshold: {best_threshold:.2f}"
)


print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

