"""
supervised models for dataset 1
Logistic regression and random forest
There are two runs, because the dataset1 has two excel sheets. It was suggested
to use the tempdata as a test set. Because I loose especially the DTI parameter
I also trained models on only the newdata set.
Thats why the code is split into 1st evaluation and 2nd evaluation
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_excel("data/Dataset_Assignment_1.xlsx", sheet_name=0)
temp_data = pd.read_excel("data/Dataset_Assignment_1.xlsx", sheet_name=1)

# 1st evaluation: Training data and test data from newdata
X = df.drop(columns=["ID", "Decision"])
X = pd.get_dummies(X, columns=["Gender", "Accommodation_Class"], drop_first=True)
y = df["Decision"].map({"Reject": 0, "Approve": 1})

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)
print(f"Number of Training Data {X_train1.shape[0]}")
print(f"Number of Test Data {y_test1.shape[0]}")
ratio1 = y_train1.value_counts(normalize=True) * 100
print("Ration of Approve/Reject in Training Data: ", ratio1)

# 2nd evaluation: Training data -> newdata , Test data -> tempdata
X_train2 = df.drop(columns=["ID", "DTI", "Gender", "Accommodation_Class", "Decision"])
y_train2 = df["Decision"].map({"Reject": 0, "Approve": 1})

X_test2 = temp_data.drop(
    columns=["Applicant ID", "Tier of City", "Years in Address", "Decision"]
)
X_test2.rename(columns={"Old EMI amount": "OldEmi"}, inplace=True)
X_test2 = X_test2[X_train2.columns]
y_test2 = temp_data["Decision"].map({"Reject": 0, "Approve": 1})
print(f"Number of Training Data {X_train2.shape[0]}")
print(f"Number of Test Data {y_test2.shape[0]}")
ratio2 = y_train2.value_counts(normalize=True) * 100
print("Ration of Approve/Reject in Training Data: ", ratio2)


def run_model(X_train, X_test, y_train, y_test, run):
    """
    runs the logistic regression and random forest

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression model
    log_model = LogisticRegression(
        random_state=123, class_weight="balanced", max_iter=1000
    )
    log_model.fit(X_train_scaled, y_train)
    y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
    threshold_log = 0.5
    y_pred_log = (y_prob_log >= threshold_log).astype(int)
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=123
    )
    rf_model.fit(X_train, y_train)  # not scaled
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    threshold_rf = 0.5
    y_pred_rf = (y_prob_rf >= threshold_rf).astype(int)

    def conf_matrix(y_pred, model):
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"------{model}---------")
        print(cm)
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall (TPR): {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        print("---------------")

    conf_matrix(y_pred_log, "Logistic Regression")
    conf_matrix(y_pred_rf, "Random Forest")

    fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_prob_log)
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_log, tpr_log, label="Logistic Regression")
    plt.plot(fpr_rf, tpr_rf, label="Random Forest")
    plt.plot()
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pictures/roc_run{run}.pdf", bbox_inches="tight")
    plt.show()

    def benchmark(fpr, tpr, model):
        print(f"------{model}---------")
        auc_val = auc(fpr, tpr)
        print(f"AUC: {auc_val:.2f}")
        gini = 2 * auc_val - 1
        print(f"Gini: {gini:.2f}")
        print("---------------")

    benchmark(fpr_log, tpr_log, "Logistic Regressiono")
    benchmark(fpr_rf, tpr_rf, "Random Forest")

    # additional random forest analysis
    feature_names = X_train.columns.tolist()
    feature_names = [
        (
            "Gender"
            if col.startswith("Gender")
            else "AC" if col.startswith("Accommodation_Class") else col
        )
        for col in feature_names
    ]
    importance = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    print("Importance Random forest")
    print(importance)
    importance.plot(kind="bar")
    plt.ylabel("Importance")
    plt.savefig(f"pictures/importantce_run{run}.pdf", bbox_inches="tight")
    plt.show()


print("-----------1st experiment-----------")
run_model(X_train1, X_test1, y_train1, y_test1, run="1")
print("-----------------------")
print("-----------2nd experiment-----------")
run_model(X_train2, X_test2, y_train2, y_test2, run="2")
print("-----------------------")
