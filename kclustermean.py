import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)
n = 1000

###############################################################################
# DATASET A: LOAN APPROVAL
###############################################################################
credit_score = np.clip(np.round(np.random.normal(680, 70, n)), 300, 850)
income       = np.round(np.random.lognormal(np.log(55000), 0.5, n))
dti          = np.round(np.clip(np.random.normal(0.32, 0.12, n), 0.02, 0.95), 3)
emp_years    = np.round(np.maximum(np.random.normal(7, 5, n), 0), 1)

lp = 0.4 + 0.012*(credit_score-680) + 0.00002*(income-55000) - 5*(dti-0.32) + 0.12*emp_years
approved = np.random.binomial(1, 1/(1+np.exp(-lp)))

loan = pd.DataFrame({
    "approved": approved,
    "credit_score": credit_score,
    "income": income,
    "dti": dti,
    "emp_years": emp_years
})

train, test = train_test_split(loan, test_size=0.3, stratify=loan["approved"], random_state=123)
X_train, y_train = train.drop("approved", axis=1), train["approved"]
X_test,  y_test  = test.drop("approved", axis=1), test["approved"]

# Logistic Regression
log_m = LogisticRegression(max_iter=500)
log_m.fit(X_train, y_train)
log_pb = log_m.predict_proba(X_test)[:,1]
log_pred = (log_pb > 0.5).astype(int)

# Decision Tree (tuned cp = min_impurity_decrease)
param_grid = {"min_impurity_decrease": np.linspace(0.0001, 0.01, 15)}
tree_cv = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10)
tree_cv.fit(X_train, y_train)
tree = tree_cv.best_estimator_
tree_pb = tree.predict_proba(X_test)[:,1]
tree_pred = tree.predict(X_test)

# Random Forest (tuned mtry = max_features)
param_grid = {"max_features": [1,2,3,4]}
rf_cv = GridSearchCV(RandomForestClassifier(n_estimators=500), param_grid, cv=10)
rf_cv.fit(X_train, y_train)
rf = rf_cv.best_estimator_
rf_pb = rf.predict_proba(X_test)[:,1]
rf_pred = rf.predict(X_test)

print("\n========== LOAN APPROVAL ==========")
print("Tuned cp =", tree_cv.best_params_["min_impurity_decrease"],
      "| Tuned mtry =", rf_cv.best_params_["max_features"])

print("\n-- Logistic --\n", confusion_matrix(y_test, log_pred))
print("\n-- Tree --\n", confusion_matrix(y_test, tree_pred))
print("\n-- Random Forest --\n", confusion_matrix(y_test, rf_pred))

print("\nAUC -> Logistic: %.3f | Tree: %.3f | RF: %.3f" %
      (roc_auc_score(y_test, log_pb),
       roc_auc_score(y_test, tree_pb),
       roc_auc_score(y_test, rf_pb)))

# Bootstrapped RF accuracy
accs = []
for _ in range(300):
    s = np.random.choice(len(y_test), replace=True, size=len(y_test))
    accs.append(np.mean(rf_pred[s] == y_test.values[s]))

print("RF bootstrapped accuracy 95% CI:",
      np.quantile(accs, 0.025), np.quantile(accs, 0.975))

# ROC Plot
fpr_log, tpr_log, _ = roc_curve(y_test, log_pb)
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_pb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pb)

plt.plot(fpr_log, tpr_log, label="Logistic")
plt.plot(fpr_tree, tpr_tree, label="Tree")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1],[0,1],"--",color="gray")
plt.legend(); plt.title("ROC - Loan Approval"); plt.show()

###############################################################################
# DATASET B: CREDIT DEFAULT
###############################################################################
utilization = np.round(np.clip(np.random.uniform(0.05, 1.2, n), 0.05, 1.5), 3)
late_pmts   = np.round(np.maximum(np.random.normal(2, 1.5, n), 0))
mo_income   = np.round(np.random.lognormal(np.log(4500), 0.45, n))
num_acct    = np.round(np.maximum(np.random.normal(6, 3, n), 1))

lp = -3 + 2.5*(utilization-0.4) + 0.6*late_pmts - 0.0002*(mo_income-4500) + 0.05*(num_acct-6)
default = np.random.binomial(1, 1/(1+np.exp(-lp)))

df = pd.DataFrame({
    "default": default,
    "utilization": utilization,
    "late_pmts": late_pmts,
    "mo_income": mo_income,
    "num_acct": num_acct
})

train, test = train_test_split(df, test_size=0.3, stratify=df["default"], random_state=123)
X_train, y_train = train.drop("default", axis=1), train["default"]
X_test,  y_test  = test.drop("default", axis=1), test["default"]

# Logistic
log_m = LogisticRegression(max_iter=500)
log_m.fit(X_train, y_train)
log_pb = log_m.predict_proba(X_test)[:,1]
log_pred = (log_pb > 0.5).astype(int)

# Tree
tree_cv = GridSearchCV(DecisionTreeClassifier(),
                       {"min_impurity_decrease": np.linspace(0.0001, 0.01, 15)}, cv=10)
tree_cv.fit(X_train, y_train)
tree = tree_cv.best_estimator_
tree_pb = tree.predict_proba(X_test)[:,1]
tree_pred = tree.predict(X_test)

# Random Forest
rf_cv = GridSearchCV(RandomForestClassifier(n_estimators=500),
                     {"max_features": [1,2,3,4]}, cv=10)
rf_cv.fit(X_train, y_train)
rf = rf_cv.best_estimator_
rf_pb = rf.predict_proba(X_test)[:,1]
rf_pred = rf.predict(X_test)

print("\n========== CREDIT DEFAULT ==========")
print("Tuned cp =", tree_cv.best_params_["min_impurity_decrease"],
      "| Tuned mtry =", rf_cv.best_params_["max_features"])

print("\n-- Logistic --\n", confusion_matrix(y_test, log_pred))
print("\n-- Tree --\n", confusion_matrix(y_test, tree_pred))
print("\n-- Random Forest --\n", confusion_matrix(y_test, rf_pred))

print("\nAUC -> Logistic: %.3f | Tree: %.3f | RF: %.3f" %
      (roc_auc_score(y_test, log_pb),
       roc_auc_score(y_test, tree_pb),
       roc_auc_score(y_test, rf_pb)))

# Bootstrapping
accs = []
for _ in range(300):
    s = np.random.choice(len(y_test), replace=True, size=len(y_test))
    accs.append(np.mean(rf_pred[s] == y_test.values[s]))

print("RF bootstrapped accuracy 95% CI:",
      np.quantile(accs, 0.025), np.quantile(accs, 0.975))
