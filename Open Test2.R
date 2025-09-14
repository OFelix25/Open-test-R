
# Customer Churn Prediction - Full R Script
install.packages("randomForest")
install.packages("xgboost")
install.packages("e1071")
install.packages("rpart.plot")
install.packages("rpart")
install.packages("ROCR")
install.packages("ggplot2")

# Load required libraries
packages <- c("dplyr", "ggplot2", "caret", "corrplot", "randomForest", "xgboost")
installed <- packages %in% installed.packages()
if(any(!installed)) install.packages(packages[!installed])
lapply(packages, library, character.only = TRUE)

library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(corrplot)
library(randomForest)
library(xgboost)
library(pROC)
# Removing gglpot2
remove.packages("ggplot2")
# Then reinstall
install.packages("ggplot2")
# Loadng the gglpot2
library(ggplot2)

# 1. Loading the Dataset

churn_data <- read.csv("C:/Users/HP/Desktop/Open test 1/Test 2/Telco Churn.csv", stringsAsFactors = FALSE)

# 2. First EDA (Raw Data)
# Summary statistics
summary(churn_data[, c("tenure", "MonthlyCharges", "TotalCharges")])

# Churn distribution
ggplot(churn_data, aes(x=Churn)) + 
  geom_bar(fill="skyblue") + 
  ggtitle("Churn Distribution")

# Boxplots: charges vs churn
ggplot(churn_data, aes(x=Churn, y=MonthlyCharges, fill=Churn)) +
  geom_boxplot() + ggtitle("Monthly Charges by Churn")

# Correlation heatmap for numerical features
num_vars <- churn_data %>% 
  select(tenure, MonthlyCharges, TotalCharges) %>% 
  mutate_all(as.numeric)
corrplot(cor(num_vars, use="complete.obs"), method="color")

# 3. Preprocessing
# Handle missing values
churn_data <- churn_data %>% drop_na()

# Convert TotalCharges to numeric (it may be character)
churn_data$TotalCharges <- as.numeric(churn_data$TotalCharges)

# Removing the customerID column 
# It is an identifier, not a feature for the model.
churn_data <- churn_data %>% select(-customerID)

# One-hot encoding
dummies <- dummyVars(Churn ~ ., data = churn_data)
churn_data_encoded <- predict(dummies, newdata = churn_data) %>% as.data.frame()

# Fixing duplicate column names
names(churn_data_encoded) <- make.unique(names(churn_data_encoded))

# Adding back Churn label
churn_data_encoded$Churn <- as.factor(churn_data$Churn)

# Cleaning up column names to remove spaces and special characters
names(churn_data_encoded) <- make.names(names(churn_data_encoded), unique = TRUE)

# Normalizing only numeric features
num_cols <- sapply(churn_data_encoded, is.numeric)
preproc <- preProcess(churn_data_encoded[, num_cols], method = c("center", "scale"))
churn_data_encoded[, num_cols] <- predict(preproc, churn_data_encoded[, num_cols])

# Cleaning column names 
names(churn_data_encoded) <- make.names(names(churn_data_encoded), unique = TRUE)

# 4. Feature Engineering & Reduction
# Tenure groups
churn_data_encoded$tenure_group <- cut(churn_data$tenure,
                                       breaks=c(-Inf, 12, 48, Inf),
                                       labels=c("Short", "Medium", "Long"))

# Total charges per contract type
churn_data_encoded$total_monthly_contract <- churn_data$MonthlyCharges * as.numeric(factor(churn_data$Contract))

# CRITICAL FIX FOR PCA 
# Identify on;y the numeric columns for PCA, excluding the target 'Churn' and the new factor 'tenure_group'
# Use sapply to find numeric columns, and remove any non-numeric ones that might remain.
numeric_columns <- sapply(churn_data_encoded, is.numeric)

# Create a new dataframe with only numeric columns, and exclude 'Churn' if it's numeric
pca_features <- churn_data_encoded[, numeric_columns]

# Checking the structure of the data going into PCA
str(pca_features) 

# Running PCA on the numeric features matrix
pca_model <- prcomp(pca_features, center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca_model$x[,1:10])  # top 10 components
pca_data$Churn <- churn_data_encoded$Churn

# 5. Train-Test Split
set.seed(42)
train_index <- createDataPartition(churn_data_encoded$Churn, p=0.7, list=FALSE)
train_data <- churn_data_encoded[train_index, ]
test_data  <- churn_data_encoded[-train_index, ]

# 6. ML Modeling
# Decision Tree
library(rpart)
dt_model <- rpart(Churn ~ ., data=train_data, method="class")
dt_pred <- predict(dt_model, test_data, type="class")

# Cleaning the column names to replace spaces with dots
names(train_data) <- make.names(names(train_data), unique = TRUE)
names(test_data) <- make.names(names(test_data), unique = TRUE)
names(rf_train) <- make.names(names(rf_train), unique = TRUE)

set.seed(42)
rf_model <- randomForest(Churn ~ ., data = rf_train, importance = TRUE)

# Random Forest
# Remove customerID if present
if("customerID" %in% colnames(train_data)) {
  rf_train <- train_data %>% select(-customerID)
} else {
  rf_train <- train_data
}

set.seed(42)
rf_model <- randomForest(Churn ~ ., data=rf_train, importance=TRUE)
rf_pred <- predict(rf_model, test_data)

# XGBoost
xgb_train <- model.matrix(Churn ~ .-1, data=train_data)
y_train <- as.numeric(train_data$Churn) - 1
xgb_test  <- model.matrix(Churn ~ .-1, data=test_data)
y_test <- as.numeric(test_data$Churn) - 1

xgb_model <- xgboost(data = xgb_train, label = y_train,
                     objective = "binary:logistic", nrounds = 100,
                     verbose = 0)
xgb_pred_prob <- predict(xgb_model, xgb_test)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, "Yes", "No") %>% as.factor()

# 7. Evaluation
# Accuracy
acc_dt  <- mean(dt_pred == test_data$Churn)
acc_rf  <- mean(rf_pred == test_data$Churn)
acc_xgb <- mean(xgb_pred == test_data$Churn)
cat("Accuracy - Decision Tree:", acc_dt,
    "\nAccuracy - Random Forest:", acc_rf,
    "\nAccuracy - XGBoost:", acc_xgb, "\n")

# Confusion Matrices
confusionMatrix(dt_pred, test_data$Churn)
confusionMatrix(rf_pred, test_data$Churn)
confusionMatrix(xgb_pred, test_data$Churn)

# ROC-AUC
roc_dt  <- roc(as.numeric(test_data$Churn)-1, as.numeric(dt_pred)-1)
roc_rf  <- roc(as.numeric(test_data$Churn)-1, as.numeric(rf_pred)-1)
roc_xgb <- roc(as.numeric(test_data$Churn)-1, as.numeric(xgb_pred)-1)

cat("AUC - Decision Tree:", auc(roc_dt),
    "\nAUC - Random Forest:", auc(roc_rf),
    "\nAUC - XGBoost:", auc(roc_xgb), "\n")

# Feature importance (Random Forest)
varImpPlot(rf_model, main="Random Forest Feature Importance")

# PCA variance explained
plot(cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2)), type="b",
     xlab="Number of Components", ylab="Cumulative Variance Explained",
     main="PCA Variance Explained")

# 8. Interpretation
cat("Top Churn Drivers: check Random Forest importance plot\n")
cat("Retention Strategies: target high-risk customers, improve contract & service options.\n")

