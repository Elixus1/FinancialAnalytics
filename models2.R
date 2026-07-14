# =============================================================
# models2.R
# R-Uebersetzung von models2.py by Claude AI
# =============================================================

library(readxl)      # Excel einlesen
library(dplyr)        # Datenmanipulation
library(caret)         # createDataPartition (stratifizierter Split), Metriken
library(randomForest)   # Random Forest
library(rpart)            # Decision Tree
library(rpart.plot)       # Decision Tree Plot
library(xgboost)           # XGBoost
library(pROC)               # ROC / AUC
library(ggplot2)             # Plots

dir.create("pictures", showWarnings = FALSE)

# -------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------

conf_matrix <- function(y_test, y_pred, model) {
  cm <- table(factor(y_test, levels = c(0, 1)), factor(y_pred, levels = c(0, 1)))
  tn <- cm[1, 1]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  tp <- cm[2, 2]

  accuracy  <- (tp + tn) / (tp + tn + fp + fn)
  precision <- tp / (tp + fp)
  recall    <- tp / (tp + fn)
  f1        <- 2 * (precision * recall) / (precision + recall)

  cat(sprintf("------%s---------\n", model))
  print(cm)
  cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
  cat(sprintf("Precision: %.2f%%\n", precision * 100))
  cat(sprintf("Recall (TPR): %.2f%%\n", recall * 100))
  cat(sprintf("F1 Score: %.2f%%\n", f1 * 100))
  cat("---------------\n")

  list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1)
}

threshold_analysis <- function(y_test, y_prob, model_name = "Model") {
  # Sucht den besten Threshold basierend auf F1-Score

  thresholds <- seq(0.01, 0.9, length.out = 100)
  accuracies <- numeric(length(thresholds))
  precisions <- numeric(length(thresholds))
  recalls    <- numeric(length(thresholds))
  f1_scores  <- numeric(length(thresholds))

  for (i in seq_along(thresholds)) {
    t <- thresholds[i]
    y_pred <- as.integer(y_prob >= t)

    tp <- sum(y_pred == 1 & y_test == 1)
    tn <- sum(y_pred == 0 & y_test == 0)
    fp <- sum(y_pred == 1 & y_test == 0)
    fn <- sum(y_pred == 0 & y_test == 1)

    accuracies[i] <- (tp + tn) / (tp + tn + fp + fn)
    precisions[i] <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recalls[i]    <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1_scores[i]  <- if ((precisions[i] + recalls[i]) == 0) 0 else
      2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
  }

  plot_df <- data.frame(
    threshold = rep(thresholds, 4),
    value = c(accuracies, precisions, recalls, f1_scores),
    metric = rep(c("Accuracy", "Precision", "Recall", "F1 Score"), each = length(thresholds))
  )

  p <- ggplot(plot_df, aes(x = threshold, y = value, color = metric)) +
    geom_line() +
    labs(x = "Threshold", y = "a.u", color = NULL) +
    theme_minimal(base_size = 14)

  ggsave(filename = sprintf("pictures/thresholds_%s.pdf", model_name),
         plot = p, width = 8, height = 5, bg = "white")
  print(p)

  best_index <- which.max(f1_scores)
  cat("Best threshold based on F1:\n")
  cat(sprintf("Threshold: %.3f\n", thresholds[best_index]))
  cat(sprintf("F1 Score: %.3f\n", f1_scores[best_index]))
  cat(sprintf("Precision: %.3f\n", precisions[best_index]))
  cat(sprintf("Recall: %.3f\n", recalls[best_index]))
  cat(sprintf("Accuracy: %.3f\n", accuracies[best_index]))
}

benchmark <- function(roc_obj, model) {
  cat(sprintf("------%s---------\n", model))
  auc_val <- as.numeric(auc(roc_obj))
  cat(sprintf("AUC: %.2f\n", auc_val))
  gini <- 2 * auc_val - 1
  cat(sprintf("Gini: %.2f\n", gini))
  cat("---------------\n")
}

# -------------------------------------------------------------
# Daten einlesen
# -------------------------------------------------------------

df <- read_excel("data/Assignment_Dataset_2.xlsx", sheet = 1)
features <- c("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI")

X <- as.matrix(df[, features])
Y <- ifelse(df$Manipulater == "Yes", 1, 0)

# -------------------------------------------------------------
# Train/Test Split (stratifiziert, analog train_test_split mit stratify=Y)
# -------------------------------------------------------------

set.seed(123)  # gleiche Samples im ganzen Skript
train_index <- createDataPartition(Y, p = 0.8, list = FALSE)

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- Y[train_index]
y_test  <- Y[-train_index]

# -------------------------------------------------------------
# Skalierung
# -------------------------------------------------------------

scaler_center <- apply(X_train, 2, mean)
scaler_scale  <- apply(X_train, 2, sd)

X_train_scaled <- scale(X_train, center = scaler_center, scale = scaler_scale)
X_test_scaled  <- scale(X_test, center = scaler_center, scale = scaler_scale)

# -------------------------------------------------------------
# Logistische Regression (class_weight="balanced" -> inverse Gewichte)
# -------------------------------------------------------------

class_freq <- table(y_train)
weights_log <- ifelse(y_train == 1,
                       unname(length(y_train) / (2 * class_freq["1"])),
                       unname(length(y_train) / (2 * class_freq["0"])))

log_data <- data.frame(X_train_scaled)
log_data$y <- y_train

log_model <- glm(y ~ ., data = log_data, family = binomial(),
                  weights = weights_log)

test_data_log <- data.frame(X_test_scaled)
y_prob_log <- predict(log_model, newdata = test_data_log, type = "response")
threshold_log <- 0.5
y_pred_log <- as.integer(y_prob_log >= threshold_log)
log_res <- conf_matrix(y_test, y_pred_log, "Logistic Regression")

# -------------------------------------------------------------
threshold_analysis(y_test, y_prob_log, "Logistic Regression")

# -------------------------------------------------------------
# Random Forest
# -------------------------------------------------------------
# GridSearchCV wurde bereits offline durchgefuehrt, bestes Ergebnis:
# max_depth=5 (~ als maxnodes/Tiefenbegrenzung in R nicht 1:1 uebertragbar,
# daher ueber nodesize/maxnodes angenaehert), min_samples_leaf=5,
# min_samples_split=2, n_estimators=500

rf_data <- data.frame(X_train)
rf_data$y <- factor(y_train, levels = c(0, 1))

# class_weight="balanced" -> classwt umgekehrt proportional zu Klassenhaeufigkeit
# unname() ist noetig, da class_freq["0"]/["1"] selbst schon Namen tragen und
# c("0" = class_freq["0"], ...) sonst verschachtelte Namen wie "0.0" erzeugt,
# die randomForest nicht als gueltige Klassenlabel erkennt
classwt <- c("0" = unname(length(y_train) / (2 * class_freq["0"])),
             "1" = unname(length(y_train) / (2 * class_freq["1"])))

set.seed(123)
rf_model <- randomForest(
  y ~ .,
  data = rf_data,
  ntree = 500,
  nodesize = 5,       # ~ min_samples_leaf
  classwt = classwt,
  maxnodes = 2^5       # ~ Begrenzung analog max_depth=5
)

y_prob_rf <- predict(rf_model, newdata = data.frame(X_test), type = "prob")[, "1"]
threshold_rf <- 0.5
y_pred_rf <- as.integer(y_prob_rf >= threshold_rf)
rf_res <- conf_matrix(y_test, y_pred_rf, "Random Forest")

# -------------------------------------------------------------
threshold_analysis(y_test, y_prob_rf, "Random Forest")

# -------------------------------------------------------------
# RF Feature Importance
# -------------------------------------------------------------

importance_vals <- importance(rf_model)[, "MeanDecreaseGini"]
importance_df <- data.frame(feature = names(importance_vals), importance = importance_vals)
importance_df <- importance_df[order(-importance_df$importance), ]

cat("Importance Random forest\n")
print(importance_df)

p_imp <- ggplot(importance_df, aes(x = reorder(feature, -importance), y = importance)) +
  geom_col() +
  labs(x = NULL, y = "Importance") +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("pictures/rf_feature_importance2.pdf", plot = p_imp, bg = "white")
print(p_imp)

# -------------------------------------------------------------
# Decision Tree
# -------------------------------------------------------------

dt_data <- data.frame(X_train)
dt_data$y <- factor(y_train, levels = c(0, 1))

# class_weight="balanced" ueber Prior-Wahrscheinlichkeiten angenaehert
prior_0 <- 0.5
prior_1 <- 0.5

dt_model <- rpart(
  y ~ .,
  data = dt_data,
  method = "class",
  parms = list(prior = c(prior_0, prior_1)),
  control = rpart.control(maxdepth = 5, minsplit = 2, minbucket = 5)
)

y_prob_dt <- predict(dt_model, newdata = data.frame(X_test), type = "prob")[, "1"]
threshold_dt <- 0.5
y_pred_dt <- as.integer(y_prob_dt >= threshold_dt)
dt_res <- conf_matrix(y_test, y_pred_dt, "Decision Tree")

# -------------------------------------------------------------
threshold_analysis(y_test, y_prob_dt, "Decision Tree")

# -------------------------------------------------------------
pdf("pictures/decision_tree.pdf")
rpart.plot(dt_model, extra = 104, fallen.leaves = TRUE, box.palette = "auto")
dev.off()

# -------------------------------------------------------------
# XGBoost
# -------------------------------------------------------------
# GridSearchCV wurde bereits offline durchgefuehrt, bestes Ergebnis:
# colsample_bytree=0.8, learning_rate=0.1, max_depth=4,
# min_child_weight=5, n_estimators=600, subsample=0.8
# Bester F1: 0.4488311688311688

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 4,
  eta = 0.1,               # entspricht learning_rate
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = 1200 / 39
)

set.seed(123)
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 600           # entspricht n_estimators
)

y_prob_xgb <- predict(xgb_model, dtest)
threshold_xgb <- 0.5
y_pred_xgb <- as.integer(y_prob_xgb >= threshold_xgb)
xgb_res <- conf_matrix(y_test, y_pred_xgb, "XGBoost")

# -------------------------------------------------------------
threshold_analysis(y_test, y_prob_xgb, "XGBoost")

# -------------------------------------------------------------
# ROC Kurven
# -------------------------------------------------------------

roc_log <- roc(y_test, y_prob_log, quiet = TRUE)
roc_rf  <- roc(y_test, y_prob_rf, quiet = TRUE)
roc_dt  <- roc(y_test, y_prob_dt, quiet = TRUE)
roc_xgb <- roc(y_test, y_prob_xgb, quiet = TRUE)

roc_df <- rbind(
  data.frame(fpr = 1 - roc_log$specificities, tpr = roc_log$sensitivities, model = "Logistic Regression"),
  data.frame(fpr = 1 - roc_rf$specificities, tpr = roc_rf$sensitivities, model = "Random Forest"),
  data.frame(fpr = 1 - roc_dt$specificities, tpr = roc_dt$sensitivities, model = "Decision Tree"),
  data.frame(fpr = 1 - roc_xgb$specificities, tpr = roc_xgb$sensitivities, model = "XGBoost")
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(x = "False Positive Rate", y = "True Positive Rate", color = NULL) +
  coord_equal() +
  theme_minimal(base_size = 14)

ggsave("pictures/roc_2.pdf", plot = p_roc, width = 6, height = 6, bg = "white")
print(p_roc)

benchmark(roc_log, "Logistic Regression")
benchmark(roc_dt, "Decision Tree")
benchmark(roc_rf, "Random Forest")
benchmark(roc_xgb, "XGBoost")
