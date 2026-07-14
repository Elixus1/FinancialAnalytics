# =====================================================
# Supervised models for Dataset 1
# Logistic Regression and Random Forest
# =====================================================

library(readxl)
library(caret)
library(randomForest)
library(pROC)

set.seed(123)

# -----------------------------------------------------
# Load data
# -----------------------------------------------------
getwd()
df <- read_excel("data/Dataset_Assignment_1.xlsx", sheet = 1)
temp_data <- read_excel("data/Dataset_Assignment_1.xlsx", sheet = 2)

# -----------------------------------------------------
# 1st evaluation
# Random train/test split
# -----------------------------------------------------

X <- subset(df, select = -c(ID, Decision))

# Convert categorical variables to dummy variables
X <- model.matrix(~ . - 1, data = X)

y <- factor(df$Decision,
            levels = c("Reject", "Approve"))

train_index <- createDataPartition(y,
                                   p = 0.7,
                                   list = FALSE)

X_train1 <- X[train_index, ]
X_test1  <- X[-train_index, ]

y_train1 <- y[train_index]
y_test1  <- y[-train_index]

cat("Training observations:", nrow(X_train1), "\n")
cat("Test observations:", nrow(X_test1), "\n")

prop.table(table(y_train1))

# -----------------------------------------------------
# 2nd evaluation
# Train on Dataset 1
# Test on temporary dataset
# -----------------------------------------------------

X_train2 <- subset(df,
                   select = -c(ID,
                               DTI,
                               Gender,
                               Accommodation_Class,
                               Decision))

y_train2 <- factor(df$Decision,
                   levels = c("Reject","Approve"))

X_test2 <- subset(temp_data,
                  select = -c(`Applicant ID`,
                              `Tier of City`,
                              `Years in Address`,
                              Decision))

names(X_test2)[names(X_test2)=="Old EMI amount"] <- "OldEmi"

X_test2 <- X_test2[, names(X_train2)]

y_test2 <- factor(temp_data$Decision,
                  levels = c("Reject","Approve"))

cat("Training observations:", nrow(X_train2), "\n")
cat("Test observations:", nrow(X_test2), "\n")

prop.table(table(y_train2))

# =====================================================
# Function to run both models
# =====================================================

run_model <- function(X_train,
                      X_test,
                      y_train,
                      y_test,
                      run){
  
  # ---------------------------------------------------
  # Standardization
  # ---------------------------------------------------
  
  scaler <- preProcess(X_train,
                       method = c("center","scale"))
  
  X_train_scaled <- predict(scaler, X_train)
  X_test_scaled <- predict(scaler, X_test)
  
  # ---------------------------------------------------
  # Logistic Regression
  # ---------------------------------------------------
  
  train_log <- data.frame(X_train_scaled,
                          Decision = y_train)
  
  log_model <- glm(
    Decision ~ .,
    data = train_log,
    family = binomial()
  )
  
  prob_log <- predict(
    log_model,
    newdata = X_test_scaled,
    type = "response"
  )
  
  pred_log <- ifelse(prob_log >= 0.5,
                     "Approve",
                     "Reject")
  
  pred_log <- factor(pred_log,
                     levels = levels(y_test))
  
  # ---------------------------------------------------
  # Random Forest
  # ---------------------------------------------------
  
  train_rf <- data.frame(X_train,
                         Decision = y_train)
  
  rf_model <- randomForest(
    Decision ~ .,
    data = train_rf,
    ntree = 100,
    importance = TRUE
  )
  
  prob_rf <- predict(
    rf_model,
    newdata = X_test,
    type = "prob")[,"Approve"]
  
  pred_rf <- ifelse(prob_rf >= 0.5,
                    "Approve",
                    "Reject")
  
  pred_rf <- factor(pred_rf,
                    levels = levels(y_test))
  
  # ---------------------------------------------------
  # Performance metrics
  # ---------------------------------------------------
  
  evaluate_model <- function(pred,
                             truth,
                             model_name){
    
    cm <- confusionMatrix(pred,
                          truth,
                          positive = "Approve")
    
    cat("\n----------------------------\n")
    cat(model_name,"\n")
    cat("----------------------------\n")
    
    print(cm$table)
    
    cat("Accuracy:",
        round(cm$overall["Accuracy"],3),"\n")
    
    cat("Precision:",
        round(cm$byClass["Pos Pred Value"],3),"\n")
    
    cat("Recall:",
        round(cm$byClass["Sensitivity"],3),"\n")
    
    cat("F1:",
        round(cm$byClass["F1"],3),"\n")
    
  }
  
  evaluate_model(pred_log,
                 y_test,
                 "Logistic Regression")
  
  evaluate_model(pred_rf,
                 y_test,
                 "Random Forest")
  
  # ---------------------------------------------------
  # ROC curves
  # ---------------------------------------------------
  
  roc_log <- roc(y_test,
                 prob_log)
  
  roc_rf <- roc(y_test,
                prob_rf)
  
  plot(roc_log,
       col = "blue",
       main = "ROC Curve")
  
  lines(roc_rf,
        col = "red")
  
  legend("bottomright",
         legend = c("Logistic Regression",
                    "Random Forest"),
         col = c("blue","red"),
         lwd = 2)
  
  # ---------------------------------------------------
  # AUC / Gini
  # ---------------------------------------------------
  
  cat("\nLogistic Regression\n")
  cat("AUC:", auc(roc_log), "\n")
  cat("Gini:", 2*auc(roc_log)-1,"\n")
  
  cat("\nRandom Forest\n")
  cat("AUC:", auc(roc_rf), "\n")
  cat("Gini:", 2*auc(roc_rf)-1,"\n")
  
  # ---------------------------------------------------
  # Variable Importance
  # ---------------------------------------------------
  
  importance <- importance(rf_model)
  
  print(importance)
  
  varImpPlot(rf_model)
  
}

# =====================================================
# Run experiments
# =====================================================

cat("\n=========== Experiment 1 ===========\n")

run_model(
  X_train1,
  as.data.frame(X_test1),
  y_train1,
  y_test1,
  run = 1
)

cat("\n=========== Experiment 2 ===========\n")

run_model(
  X_train2,
  X_test2,
  y_train2,
  y_test2,
  run = 2
)

