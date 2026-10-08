import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

df=pd.read_csv("data/ai4i2020.csv")

df.drop(["UDI","Product ID"],axis=1,inplace=True)

df["Power"]=df["Torque [Nm]"]*df["Rotational speed [rpm]"]

le=LabelEncoder()

df["Type"]=le.fit_transform(df["Type"])

X=df.drop("Machine failure",axis=1)

y=df["Machine failure"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X_train,y_train)

probs=model.predict_proba(X_test)[:,1]

def calculate_cost(y_true,predictions):
    tn,fp,fn,tp=confusion_matrix(y_true,predictions).ravel()
    cost=(fn*5000)+(fp*500)
    return cost,tn,fp,fn,tp

pred_default=(probs>=0.5).astype(int)
cost_default,tn,fp,fn,tp=calculate_cost(y_test,pred_default)

print("="*50)
print("BASELINE (Threshold = 0.5)")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Financial Loss: ${cost_default:,}")
print("="*50)
best_thresh=0.5

best_cost=float("inf")

best_preds= None
thresholds=np.arange(0.01,0.99,0.01)

costs=[]

for thresh in thresholds:
    preds=(probs>= thresh).astype(int)
    cost,_,_,_,_=calculate_cost(y_test,preds)

    costs.append(cost)

    if cost<best_cost:
        best_cost=cost
        best_thresh=thresh
        best_preds=preds

tn2,fp2,fn2,tp2= confusion_matrix(y_test,best_preds).ravel()

print("="*50)
print("YOUR OPTIMIZED MODEL (threshold = (:.2f))".format(best_thresh))
print(f"Confusion Matrix: TN={tn2},FP={fp2},FN={fn2},TP={tn2}")
print(f"Total Financial Loss: ${best_cost:,}")
print(f"Savings: ${cost_default - best_cost:,} REDUCED")
print(f"PERCENTAGE IMPROVEMENT: {((cost_default-best_cost)/cost_default)*100:.1f}%")
print("="*50)

plt.figure(figsize=(10,6))
plt.plot(thresholds,costs,linewidth=2,label="Total Cost")
plt.axvline(x=best_thresh,color="green",linestyle="--",label=f"Optimal Threshold ({best_thresh:.2f})")
plt.xlabel("Decision Threshold")
plt.ylabel("Total Financial Loss($)")
plt.title("Cost-Sensitive Threshold Optimization for Machine Failure Prediction")
plt.legend()
plt.grid(True)
plt.savefig("threshold_optimization.png",dpi=300)
plt.show()
