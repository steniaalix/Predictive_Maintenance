import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

df=pd.read_csv("data/ai4i2020.csv")
df.drop(['UDI','Product ID'],axis=1,inplace=True)

df['Power']=df['Torque [Nm]']*df["Rotational speed [rpm]"]

le=LabelEncoder()

df['Type']=le.fit_transform(df['Type'])

target="TWF"

failure_cols=["Machine failure","HDF","PWF","OSF","RNF"]
X=df.drop(columns=failure_cols +[target])
y=df[target]

X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training set:",X_train.shape)
print("Validation set:",X_val.shape)
print("Test set:",X_test.shape)

print("\n TWF distribution:")
print("Train:")
print(y_train.value_counts())

print("\nValidation:")
print(y_val.value_counts())


print("\n Test:")
print(y_test.value_counts())

model=DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train,y_train)

for feature,importance in zip(X.columns,model.feature_importances_):
    print(feature,importance)

    
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


val_probs=model.predict_proba(X_val)[:,1]
test_probs=model.predict_proba(X_test)[:,1]

evaluate_model(
    y_val,
    val_probs,
    threshold=0.5,
    name="VALIDATION SET - DEFAULT MODEL"
)

evaluate_model(
    y_test,
    test_probs,
    threshold=0.5,
    name="TEST SET - DEFAULT MODEL"
)


FN_COST=5000
FP_COST=500

def calculate_cost(y_true,preds):
    tn,fp,fn,tp=confusion_matrix(
        y_true,
        preds,
        labels=[0,1]
    ).ravel()

    total_cost=(fn*FN_COST)+(fp*FP_COST)

    return total_cost,tn,fp,fn,tp

thresholds= np.arange(0.01,1.00,0.01)

validation_costs=[]

best_threshold= 0.5
best_cost=float("inf")

for threshold in thresholds:
    val_preds=(val_probs>=threshold).astype(int)
    cost,tn,fp,fn,tp=calculate_cost(y_val,val_preds)

    validation_costs.append(cost)

    if cost< best_cost:
        best_cost=cost
        best_threshold=threshold

print("\n" + "=" * 50)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 50)

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Minimum Validation Cost: ${best_cost:,}")

# ==========================================
# FINAL TEST SET EVALUATION
# ==========================================

optimized_threshold = best_threshold

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
    validation_costs,
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
plt.ylabel("Validation Cost ($)")
plt.title("Cost-Aware Threshold Optimization for Tool Wear Failure")
plt.legend()
plt.grid(True)
