import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_predict


df=pd.read_csv("data/ai4i2020.csv")
df.drop(['UDI','Product ID'],axis=1,inplace=True)

df['Power']=df['Torque [Nm]']*df["Rotational speed [rpm]"]

le=LabelEncoder()

df['Type']=le.fit_transform(df['Type'])

target="PWF"

failure_cols=["Machine failure","HDF","TWF","OSF","RNF"]
X=df.drop(columns=failure_cols +[target])
y=df[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


print("Training set:",X_train.shape)
print("Test set:",X_test.shape)

print("\n HDF distribution:")
print("Train:")
print(y_train.value_counts())



print("\n Test:")
print(y_test.value_counts())

model=RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

param_grid={
    "n_estimators":[200,300,500],
    "max_depth":[None,3,5,10],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2,5,10],
    "max_features":["sqrt","log2",None]
}
grid=GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)
grid.fit(X_train,y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV F1: {grid.best_score_}")
final_model=grid.best_estimator_

cv_model=RandomForestClassifier(
    **grid.best_params_,
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
    n_jobs=-1
)[:,1]



def evaluate_model(y_true,probs,threshold=0.5,name="Model"):
    preds=(probs>=threshold).astype(int)

    tn,fp,fn,tp=confusion_matrix(
        y_true,
        preds,labels=[0,1]
    ).ravel()

    print("\n"+ "="*50)
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

    return preds, (tn,fp,fn,tp)





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
        train_probs_cv>=threshold
    ).astype(int)

    cost,tn,fp,fn,tp= calculate_cost(
        y_train,
        cv_preds
    )

    cv_costs.append(cost)

    if cost< best_cost:
        best_cost=cost
        best_threshold=threshold


print("\n"+"="*50)
print("CROSS-VALIDATED THRESHOLD OPTIMIZATION")
print("="*50)

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Minimum Cross-Validated Cost: ${best_cost:,}")

# ==========================================
# FINAL TEST SET EVALUATION
# ==========================================

optimized_threshold = best_threshold

test_probs=final_model.predict_proba(X_test)[:,1]


evaluate_model(
    y_test,
    test_probs,
    threshold=0.5,
    name="TEST SET - DEFAULT MODEL"
)


evaluate_model(
    y_test,test_probs,
    threshold=best_threshold,
    name="TEST SET - COST-OPTIMIZED THRESHOLD"
)


# Default predictions
test_preds_default = (test_probs >= 0.5).astype(int)




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



plt.figure(figsize=(10,6))

plt.plot(
    thresholds,
    cv_costs,
    linewidth=2
)

plt.axvline(
    x=0.5,
    linestyle="--",
    label="Default Threshold (0.50)"
)

plt.axvline(
    x=best_threshold,
    linestyle="--",
    label=f"Optimal Threshold ({best_threshold:.2f})"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Cross-Validation Cost ($)")
plt.title("Cross-validated Cost-Aware Threshold Optimization for Power Failure")
plt.legend()
plt.grid(True)

plt.show()


artifact={
    "model":final_model,
    "encoder":le,
    "threshold":best_threshold,
    "feature_order":list(X_train.columns),
    "sklearn_version":__import__("sklearn").__version__,
}

joblib.dump(artifact,"app/pwf_model.joblib")

