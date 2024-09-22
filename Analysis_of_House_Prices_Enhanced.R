# Install and load necessary packages
required_packages <- c("GGally", "ggplot2", "dplyr", "glmnet", "randomForest", "caret", "iml", "car", "xgboost", "e1071", "MASS", "reshape2")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load libraries quietly
suppressMessages(lapply(required_packages, require, character.only = TRUE))

# Load the data
house_data <- read.csv("real-estate-taiwan.csv")

# Initial Data Exploration
print("Initial Data Overview:")
print(summary(house_data))
print(str(house_data))

# Data Exploration and Visualization
print("Exploratory Data Analysis:")

# Correlation Matrix
correlation_matrix <- cor(house_data[, sapply(house_data, is.numeric)])
print(correlation_matrix)

# Heatmap of correlations
corr_melt <- reshape2::melt(correlation_matrix)
print(ggplot(corr_melt, aes(Var1, Var2, fill = value)) +
        geom_tile() +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = "Heatmap of Feature Correlations"))

# Pairwise relationships
print("Pairwise Plot:")
print(ggpairs(house_data))

# Boxplot: House Price by Convenience Stores
print(ggplot(house_data, aes(x = as.factor(convenience_stores), y = house_price)) +
        geom_boxplot(fill = "lightblue") +
        theme_minimal() +
        labs(title = "House Price by Number of Convenience Stores",
             x = "Number of Convenience Stores", y = "House Price (in millions)"))

# Histogram of House Price
print(ggplot(house_data, aes(house_price)) +
        geom_histogram(bins = 30, fill = "blue", color = "white") +
        labs(title = "Distribution of House Prices", x = "House Price", y = "Frequency"))

# Feature Engineering
house_data$log_distance <- log(house_data$mrt_distance + 1)
house_data$age_distance_interaction <- house_data$house_age * house_data$mrt_distance
house_data$house_age_sq <- house_data$house_age^2
house_data$mrt_distance_sq <- house_data$mrt_distance^2

# Standardize features
house_data_std <- house_data
numeric_cols <- sapply(house_data_std, is.numeric)
house_data_std[numeric_cols] <- scale(house_data_std[numeric_cols])

# Prepare data for modeling
x <- model.matrix(house_price ~ house_age + mrt_distance + convenience_stores + latitude + longitude + log_distance + age_distance_interaction + house_age_sq + mrt_distance_sq - 1, house_data_std)
y <- house_data_std$house_price

set.seed(123)
train_index <- createDataPartition(house_data_std$house_price, p = 0.8, list = FALSE)
train_data <- house_data_std[train_index, ]
test_data <- house_data_std[-train_index, ]

x_train <- x[train_index, ]
x_test <- x[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Function to evaluate model performance
evaluate_model <- function(model, test_data, y_test, model_type = "regression") {
  pred <- predict(model, newdata = test_data)
  r_squared <- cor(pred, y_test)^2
  print(paste(model_type, "R-squared on Test Set:", r_squared))
  return(r_squared)
}

# Linear Regression Model
linear_model <- lm(house_price ~ house_age + mrt_distance + convenience_stores + latitude + longitude + log_distance + age_distance_interaction + house_age_sq + mrt_distance_sq, data = train_data)
summary(linear_model)

# Ridge Regression Model with cross-validation
ridge_model <- cv.glmnet(as.matrix(x_train), y_train, alpha = 0)

# Create x_test matrix
x_test <- model.matrix(house_price ~ house_age + mrt_distance + convenience_stores + latitude + longitude + log_distance + age_distance_interaction + house_age_sq + mrt_distance_sq - 1, test_data)

# Ensure x_test is a matrix
if (!is.matrix(x_test)) stop("x_test is not a matrix")

# Predict and calculate R-squared
ridge_pred <- as.vector(predict(ridge_model, s = ridge_model$lambda.min, newx = x_test))
ridge_r_squared <- cor(ridge_pred, y_test)^2
print(paste("Ridge Regression R-squared:", ridge_r_squared))

# Lasso Regression Model with cross-validation
lasso_model <- cv.glmnet(as.matrix(x_train), y_train, alpha = 1)
lasso_pred <- predict(lasso_model, s = lasso_model$lambda.min, newx = as.matrix(x_test))

# Random Forest Model
rf_model <- randomForest(house_price ~ ., data = train_data, importance = TRUE)
print(rf_model)
rf_pred <- predict(rf_model, newdata = test_data)

# Support Vector Machine Model
svm_model <- svm(house_price ~ ., data = train_data)
print(svm_model)
svm_pred <- predict(svm_model, newdata = test_data)

# XGBoost Model
xgb_model <- xgboost(data = as.matrix(x_train), label = y_train, nrounds = 100, objective = "reg:squarederror")
xgb_pred <- predict(xgb_model, as.matrix(x_test))

# Model Evaluation
linear_r_squared <- evaluate_model(linear_model, test_data, test_data$house_price, "Linear Regression")
ridge_r_squared <- evaluate_model(ridge_model, x_test, y_test, "Ridge Regression")
lasso_r_squared <- evaluate_model(lasso_model, x_test, y_test, "Lasso Regression")
rf_r_squared <- evaluate_model(rf_model, test_data, test_data$house_price, "Random Forest")
svm_r_squared <- evaluate_model(svm_model, test_data, test_data$house_price, "SVM")
xgb_r_squared <- evaluate_model(xgb_model, x_test, y_test, "XGBoost")

# Feature Importance (Random Forest)
rf_predictor <- Predictor$new(rf_model, data = test_data[, -which(names(test_data) == "house_price")], y = test_data$house_price, predict.function = function(model, newdata) as.numeric(predict(model, newdata = newdata)))
feature_imp <- FeatureImp$new(rf_predictor, loss = "mse")
feature_imp$plot()

# Residual Analysis for Linear Model
residuals <- residuals(linear_model)
print(ggplot(data.frame(residuals), aes(residuals)) +
        geom_histogram(bins = 30, fill = "orange", color = "black") +
        labs(title = "Residuals Distribution", x = "Residuals", y = "Frequency"))

# Residual vs Fitted Plot
print(ggplot(data.frame(fitted = fitted(linear_model), residuals = residuals), aes(fitted, residuals)) +
        geom_point(color = "blue") +
        geom_smooth(se = FALSE, color = "red") +
        labs(title = "Residuals vs Fitted", x = "Fitted values", y = "Residuals"))

# Variance Inflation Factor (VIF) for Linear Model
vif_values <- vif(linear_model)
print("VIF for Linear Model:")
print(vif_values)

# Cross-validation for model evaluation (Linear Model)
train_control <- trainControl(method = "cv", number = 10)
linear_cv <- train(house_price ~ ., data = train_data, method = "lm", trControl = train_control)
print("Cross-Validation Results for Linear Model:")
print(linear_cv)

