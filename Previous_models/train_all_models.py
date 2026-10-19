import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict
)
from sklearn.metrics import confusion_matrix

df=pd.read_csv("data/ai4i2020.csv")
df.drop(["UDI","Product ID"],axis=1,inplace=True)

df["Power"]=df['Torque [Nm]']* df["Rotational speed [rpm]"]

le=LabelEncoder()

df["Type"]=le.fit_transform(df["Type"])


failure_targets=["TWF","HDF","PWF","OSF","RNF"]

drop_colums=["Machine failure","TWF","HDF","PWF","OSF","RNF"]

X=df.drop(columns=drop_colums)


train_indices,test_indices=train_test_split(
    df.index,
    test_size=0.3,
    random_state=42
)

X_train=X.loc[train_indices]
X_test=X.loc[test_indices]

print("="*50)
print("TRAINING MULTI-FAILURE PREDICTION SYSTEM")
print("="*60)

print("Taining samples: ",len(X_train))
print("Test samples: ",len(X_test))


base_model=RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

param_grid={
    "n_estimators":[200,300,500],
    "max_depth":[5,10,None],
    "min_samples_split":[2,5],
    "min_samples_leaf":[2,5,10],
    "max_features":["sqrt","log2"]
}


FP_COST=500

failure_costs={
    "TWF":15000,
    "HDF":15000,
    "PWF":10000,
    "OSF":10000,
    "RNF":5000
}


def calculte_cost(y_true,predictions,fn_cost):
    tn,fp,fn,tp=confusion_matrix(y_true,predictions,labels=[0,1]).ravel()

    total_cost=(fn*fn_cost+fp*FP_COST)
    return total_cost,tn,fp,fn,tp


models={}
thresholds={}
results={}

for target in failure_targets:
    print("\n")
    print("="*60)
    print(f"TRAINING MODEL FOR {target}")
    print("="*60)

    y=df[target]

    y_train=y.loc[train_indices]
    y_test=y.loc[test_indices]

    print("\nTraining destribution:")
    print(y_train.value_counts())

    grid=GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(
        X_train,
        y_train
    )

    print("\nBest parameters:")
    print(grid.best_params_)

    print("Best CV F1:",grid.best_score_)


    best_params=grid.best_params_

    cv_model=RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    train_probs_cv=cross_val_predict(
        cv_model,
        X_train,
        y_train,
        cv=5,
        method="predict_proba",
        n_jobs=-1,
    )[:,1]


    thresholds_to_test=np.arange(
        0.01,
        1.00,
        0.01
    )

    best_threshold=0.5

    best_cost=float("inf")


    for threshold in thresholds_to_test:
        predictions=(
            train_probs_cv>=threshold
        ).astype(int)


        cost,tn,fp,fn,tp=calculte_cost(
            y_train,
            predictions,
            failure_costs[target]
        )
        if cost<best_cost:
            best_cost=cost
            best_threshold=threshold

    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"Minimum CV Cost: {best_cost}")

    final_model=RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
    )

    final_model.fit(X_train,y_train)

    test_probs=final_model.predict_proba(X_test)[:,1]

    test_predictions=(test_probs>=best_threshold).astype(int)

    test_cost,tn,fp,fn,tp=calculte_cost(y_test,test_predictions,failure_costs[target])

    print("\nTEST RESULTS:")

    print("TN:",tn)
    print("FP:",fp)
    print("FN:",fn)
    print("TP:",tp)

    print(f"Test Cost: ${test_cost:,}")


    models[target]=final_model
    thresholds[target]=best_threshold
    results[target]={
            "best_parmas":best_params,
            "best_cv_f1":grid.best_score_,
            "thresholds":best_threshold,
            "test_cost":test_cost,
            "TN":tn,
            "FP":fp,
            "FN":fn,
            "TP":tp
    }


artifact={
    "models":models,
    "thresholds":thresholds,
    "encoder":le,
    "feature_order":list(X.columns),
    "failure_costs":failure_costs,
    "results":results,
    "sklearn_version":__import__("sklearn").__version__
}

joblib.dump(artifact,"app/multi_failure_model.joblib")

print("\n")
print("="*60)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("\nSaved to:")
print("app/multi_failure_model.joblib")

print("\nFINAL THRESHOLDS:")

for failure, threshold in thresholds.items():
    print(f"{failure}:{threshold:.2f}")

print("\nFINAL RESULTS:")
for failure, result in results.items():
    print("\n",failure)

    for key,value in result.items():
        print(f"{key}: {value}")
        