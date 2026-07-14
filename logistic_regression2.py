"""
Logistic Regression for dataset 1
Learns with NewData and model is tested with Tempdata
however the model can only be trained on common parameters
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

df = pd.read_excel("data/Dataset_Assignment_1.xlsx", sheet_name=0)
temp_data = pd.read_excel("data/Dataset_Assignment_1.xlsx", sheet_name=1)

X_train = df.drop(columns=["ID", "DTI", "Gender", "Accommodation_Class", "Decision"])
y_train = df["Decision"].map({"Reject": 0, "Approve": 1})

X_test = temp_data.drop(
    columns=["Applicant ID", "Tier of City", "Years in Address", "Decision"]
)
X_test.rename(columns={"Old EMI amount": "OldEmi"}, inplace=True)
X_test = X_test[X_train.columns]
y_test = temp_data["Decision"].map({"Reject": 0, "Approve": 1})
print(f"Number of Training Data {X_train.shape[0]}")
print(f"Number of Test Data {y_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_prob = log_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
print(cm)
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
tp = cm[1, 1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall (TPR): {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

auc_val = auc(fpr, tpr)
print(f"AUC: {auc_val:.2f}")

gini = 2 * auc_val - 1
print(f"Gini: {gini:.2f}")
