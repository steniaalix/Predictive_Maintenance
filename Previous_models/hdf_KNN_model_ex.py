import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,precision_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV,cross_val_predict
from sklearn.pipeline import Pipeline

df=pd.read_csv("data/ai4i2020.csv")
print(df)
le=LabelEncoder()

df.drop(["UDI","Product ID"],axis=1,inplace=True)

df['Type']=le.fit_transform(df['Type'])
df['Power']=df['Torque [Nm]']*df["Rotational speed [rpm]"]

target="HDF"

failure_cols=["Machine failure","PWF","OSF","RNF","TWF"]
X=df.drop(columns=failure_cols +[target])
y=df[target]


pipe=Pipeline([('scaler',StandardScaler()),("model",KNeighborsClassifier())])
params={
    "model__n_neighbors":[3,6,10,13,15],
    "model__weights":["uniform","distance"],
    "model__metric":['euclidean','manhattan']
}

grid=GridSearchCV(pipe,params,cv=5,scoring="f1",verbose=2,n_jobs=-1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

grid.fit(X_train,y_train)
best_params=grid.best_params_
best_score=grid.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best f1 score: {best_score}")

final_model=grid.best_estimator_

train__probs_cv=cross_val_predict(
    final_model,
    X_train,
    y_train,
    cv=10,
    method="predict_proba",
    n_jobs=-1
)[:,1]


def evaluate_model(y_true,probs,threshold=0.5,name="Model"):
    preds=(probs>=threshold).astype(int)

    tn,fp,fn,tp=confusion_matrix(y_true,preds,labels=[0,1]).ravel()

    print("\n"+"="*50)
    print(name)
    print(f"Threshold: {threshold}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print(f"\nAccuracy: {accuracy_score(y_true,preds):.4f}")
    print(f"Precision: {precision_score(y_true,preds,zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true,preds,zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true,preds,zero_division=0):.4f}")

    return preds,(tn,fp,fn,tp)



FN_COST=15000
FP_COST=500

def calculate_cost(y_true,preds):
    tn,fp,fn,tp=confusion_matrix(
        y_true,
        preds,
        labels=[0,1]
    ).ravel()

    total_cost=(fn*FN_COST)+(fp*FP_COST)

    return total_cost,tn,fp,fn,tp


thresholds=np.arange(0.01,1.0,0.01)

cv_costs=[]

best_threshold=0.5

best_cost=float("inf")

for threshold in thresholds:
    cv_preds=(
        train__probs_cv>=threshold
    ).astype(int)

    cost,tn,fp,fn,tp=calculate_cost(
        y_train,
        cv_preds
    )

    cv_costs.append(cost)

    if cost<best_cost:
        best_cost=cost
        best_threshold=threshold

print("\n"+"="*50)
print("CROSS-VALIDATED THRESHOLD OPTIMIZATION")

print("="*50)

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Minimum Cross-Validated Cost: ${best_cost:,}")


optimized_threshold=best_threshold

test_probs=final_model.predict_proba(X_test)[:,1]

evaluate_model(y_test,test_probs,threshold=0.5,name="TEST SET - DEFAULT MODEL")

evaluate_model(y_test,test_probs,threshold=best_threshold,name="TEST SET - COST-OPTIMIZED THRESHOLD")

test_preds_default=(test_probs>=0.5).astype(int)


# Optimized predictions
test_preds_optimized = (
    test_probs >= optimized_threshold
).astype(int)


# Calculate default cost
default_test_cost, tn, fp, fn, tp = calculate_cost(
    y_test,
    test_preds_default
)

print("\n" + "=" * 50)
print("TEST SET — DEFAULT THRESHOLD")
print("=" * 50)

print("Threshold: 0.50")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")
print(f"Total Cost: ${default_test_cost:,}")


# Calculate optimized cost
optimized_test_cost, tn2, fp2, fn2, tp2 = calculate_cost(
    y_test,
    test_preds_optimized
)

print("\n" + "=" * 50)
print("TEST SET — COST-OPTIMIZED THRESHOLD")
print("=" * 50)

print(f"Threshold: {optimized_threshold:.2f}")
print(f"TN = {tn2}")
print(f"FP = {fp2}")
print(f"FN = {fn2}")
print(f"TP = {tp2}")
print(f"Total Cost: ${optimized_test_cost:,}")


# Savings
savings = default_test_cost - optimized_test_cost

if default_test_cost > 0:
    improvement = (
        savings / default_test_cost
    ) * 100
else:
    improvement = 0


print("\n" + "=" * 50)
print("FINAL COST COMPARISON")
print("=" * 50)

print(f"Default Cost:   ${default_test_cost:,}")
print(f"Optimized Cost: ${optimized_test_cost:,}")
print(f"Savings:        ${savings:,}")
print(f"Cost Reduction: {improvement:.2f}%")
print("=" * 50)



