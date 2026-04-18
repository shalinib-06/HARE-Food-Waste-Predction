# ============================================================
# STEP 1: Load Data & Explore
# ============================================================
data <- read.csv("E:/r project/train.csv")
head(data)
str(data)
summary(data)

# ============================================================
# STEP 2: Install & Load Libraries
# ============================================================
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("corrplot")
#install.packages("fastshap")
#install.packages("caret")
#if (!require("kernelshap")) install.packages("kernelshap")
#if (!require("randomForest")) install.packages("randomForest")

library(ggplot2)
library(dplyr)
library(corrplot)
library(fastshap)
library(randomForest)
library(kernelshap)
library(caret)

# ============================================================
# STEP 3: Check Missing Values
# ============================================================
colSums(is.na(data))

# ============================================================
# STEP 4: Clean Numeric Columns
# ============================================================
clean_numeric <- function(x) {
  as.numeric(gsub("[^0-9.]", "", x))
}

data$meals_served    <- clean_numeric(data$meals_served)
data$kitchen_staff   <- clean_numeric(data$kitchen_staff)
data$temperature_C   <- clean_numeric(data$temperature_C)
data$humidity_percent <- clean_numeric(data$humidity_percent)
data$past_waste_kg   <- clean_numeric(data$past_waste_kg)
data$food_waste_kg   <- clean_numeric(data$food_waste_kg)

# ============================================================
# STEP 5: FIX 1 — Clean Dirty Categorical Columns
# ============================================================

# Fix staff_experience
data$staff_experience <- tolower(trimws(data$staff_experience))

data$staff_experience[data$staff_experience == "nan"] <- NA
data$staff_experience <- ifelse(is.na(data$staff_experience), "unknown", data$staff_experience)

cat("staff_experience categories:\n")
print(table(data$staff_experience))

# Fix waste_category
data$waste_category <- tolower(trimws(data$waste_category))

cat("waste_category categories:\n")
print(table(data$waste_category))

# ============================================================
# STEP 6: Remove Outliers
# ============================================================
remove_outliers <- function(df, col) {
  Q1  <- quantile(df[[col]], 0.25, na.rm = TRUE)
  Q3  <- quantile(df[[col]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  df[df[[col]] >= (Q1 - 1.5 * IQR) & df[[col]] <= (Q3 + 1.5 * IQR), ]
}

data <- remove_outliers(data, "meals_served")
data <- remove_outliers(data, "food_waste_kg")
data <- remove_outliers(data, "temperature_C")
data <- remove_outliers(data, "past_waste_kg")

cat("Rows after outlier removal:", nrow(data), "\n")
cat("Max meals_served after cleaning:", max(data$meals_served, na.rm = TRUE), "\n")

# ============================================================
# STEP 7: Extract Date Parts (before removing date column)
# ============================================================
data$day   <- as.numeric(format(as.Date(data$date), "%d"))
data$month <- as.numeric(format(as.Date(data$date), "%m"))

# ============================================================
# STEP 8: Save Cleaned Data (before removing ID/date)
# ============================================================
write.csv(data, "E:/r project/cleaned_data.csv", row.names = FALSE)
cat("Cleaned data saved.\n")

# ============================================================
# STEP 9: Drop ID and Date (no longer needed)
# ============================================================
data$ID   <- NULL
data$date <- NULL

str(data)
head(data)

# ============================================================
# STEP 10: FIX 2 — Train/Test Split FIRST
# ============================================================
set.seed(123)
train_index <- sample(1:nrow(data), 0.8 * nrow(data))
train_data  <- data[train_index, ]
test_data   <- data[-train_index, ]

cat("Train rows:", nrow(train_data), "\n")
cat("Test rows: ", nrow(test_data),  "\n")

# ============================================================
# STEP 11: Feature Engineering AFTER Split (no leakage)
# ============================================================
engineer_features <- function(df) {
  df$staff_efficiency <- df$meals_served / df$kitchen_staff
  df$demand_pressure  <- df$meals_served / (df$kitchen_staff + 1)
  df$env_score        <- df$temperature_C * df$humidity_percent
  df$event_risk       <- df$special_event * df$meals_served
  df$waste_ratio      <- df$past_waste_kg / (df$past_waste_kg + 1)
  df$is_weekend       <- ifelse(df$day_of_week %in% c(5, 6), 1, 0)
  return(df)
}

train_data <- engineer_features(train_data)
test_data  <- engineer_features(test_data)

cat("Features after engineering:", ncol(train_data), "\n")

# ============================================================
# STEP 12: EDA Plots
# ============================================================

# 1. Waste vs Meals Served
png("E:/r project/waste_vs_meals.png", width = 800, height = 600)
ggplot(train_data, aes(x = meals_served, y = food_waste_kg)) +
  geom_point(color = "blue", alpha = 0.5) +
  ggtitle("Food Waste vs Meals Served") +
  xlab("Meals Served") + ylab("Food Waste (kg)") +
  theme_minimal()
dev.off()

# 2. Waste vs Temperature
png("E:/r project/waste_vs_temperature.png", width = 800, height = 600)
ggplot(train_data, aes(x = temperature_C, y = food_waste_kg)) +
  geom_point(color = "blue", alpha = 0.5) +
  ggtitle("Food Waste vs Temperature") +
  xlab("Temperature (°C)") + ylab("Food Waste (kg)") +
  theme_minimal()
dev.off()

# 3. Event vs Non-Event Waste
png("E:/r project/event_vs_waste.png", width = 800, height = 600)
ggplot(train_data, aes(x = factor(special_event), y = food_waste_kg)) +
  geom_boxplot(fill = "orange") +
  ggtitle("Event vs Non-Event Food Waste") +
  xlab("Special Event (0 = No, 1 = Yes)") + ylab("Food Waste (kg)") +
  theme_minimal()
dev.off()

# 4. Day-wise Waste
png("E:/r project/daywise_waste.png", width = 800, height = 600)
ggplot(train_data, aes(x = factor(day_of_week), y = food_waste_kg)) +
  geom_boxplot(fill = "green") +
  ggtitle("Day-wise Food Waste") +
  xlab("Day of Week") + ylab("Food Waste (kg)") +
  theme_minimal()
dev.off()

# 5. Correlation Heatmap
numeric_data <- train_data[sapply(train_data, is.numeric)]
cor_matrix   <- cor(numeric_data, use = "complete.obs")
png("E:/r project/heatmap.png", width = 800, height = 600)
corrplot(cor_matrix, method = "color", type = "upper")
dev.off()

# 6. Distribution of Food Waste
png("E:/r project/distribution.png", width = 800, height = 600)
ggplot(train_data, aes(x = food_waste_kg)) +
  geom_histogram(fill = "blue", bins = 30) +
  ggtitle("Distribution of Food Waste") +
  theme_minimal()
dev.off()

# 7. Category vs Waste
png("E:/r project/category_waste.png", width = 800, height = 600)
ggplot(train_data, aes(x = waste_category, y = food_waste_kg)) +
  geom_boxplot(fill = "lightcoral") +
  ggtitle("Food Waste by Category") +
  theme_minimal()
dev.off()

# 8. Monthly Trend
monthly_waste <- train_data %>%
  group_by(month) %>%
  summarise(
    mean_waste = mean(food_waste_kg, na.rm = TRUE),
    sd_waste   = sd(food_waste_kg,   na.rm = TRUE)
  )

png("E:/r project/monthly_trend.png", width = 900, height = 600)
ggplot(monthly_waste, aes(x = month, y = mean_waste)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(color = "darkblue", size = 3) +
  geom_errorbar(aes(ymin = mean_waste - sd_waste,
                    ymax = mean_waste + sd_waste),
                width = 0.3, color = "gray50") +
  scale_x_continuous(breaks = 1:12,
                     labels = month.abb) +
  labs(title    = "Monthly Food Waste Trend",
       subtitle = "Error bars = ±1 SD | Based on training data only",
       x = "Month", y = "Mean Food Waste (kg)") +
  theme_minimal(base_size = 14)
dev.off()

cat("All EDA plots saved.\n")

# ============================================================
# STEP 13: Baseline Linear Regression Model
# ============================================================
model <- lm(food_waste_kg ~ ., data = train_data)
summary(model)

# Predict & Evaluate
predictions <- predict(model, test_data)
actual      <- test_data$food_waste_kg
base_pred   <- predictions   

rmse <- sqrt(mean((actual - predictions)^2))
mae  <- mean(abs(actual - predictions))
r2   <- cor(actual, predictions)^2

cat("\n--- Baseline Linear Regression ---\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAE: ", round(mae,  4), "\n")
cat("R2:  ", round(r2,   4), "\n")

# Prediction Intervals
pred_intervals <- predict(model, test_data,
                          interval = "prediction", level = 0.95)
interval_df <- data.frame(
  Actual    = actual,
  Predicted = pred_intervals[, "fit"],
  Lower     = pred_intervals[, "lwr"],
  Upper     = pred_intervals[, "upr"]
)

coverage <- mean(interval_df$Actual >= interval_df$Lower &
                   interval_df$Actual <= interval_df$Upper)
cat("Prediction Interval Coverage:", round(coverage * 100, 2), "%\n")

png("E:/r project/prediction_intervals.png", width = 900, height = 600)
ggplot(interval_df[1:50, ], aes(x = 1:50)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper),
              fill = "lightblue", alpha = 0.5) +
  geom_line(aes(y = Predicted), color = "blue", linewidth = 1) +
  geom_point(aes(y = Actual),   color = "red",  size = 2) +
  labs(title    = "Prediction Intervals vs Actual Food Waste",
       subtitle = "Blue band = 95% prediction interval | Red = Actual",
       x = "Test Sample Index", y = "Food Waste (kg)") +
  theme_minimal(base_size = 14)
dev.off()

# ============================================================
# STEP 14: Predict on New Real-World Input
# ============================================================
new_data <- data.frame(
  meals_served     = 200,
  kitchen_staff    = 10,
  past_waste_kg    = 25,
  special_event    = 1,
  waste_ratio      = 25 / (25 + 1)
)

model_simple <- lm(food_waste_kg ~ meals_served + kitchen_staff +
                     past_waste_kg + special_event + waste_ratio,
                   data = train_data)

reg_pred <- predict(model_simple, new_data)
cat("\nPredicted Food Waste (new input):", round(reg_pred, 2), "kg\n")

# ============================================================
# STEP 15: Random Forest Baseline
# ============================================================
set.seed(123)
rf_model <- randomForest(food_waste_kg ~ ., data = train_data, ntree = 500)
rf_pred  <- predict(rf_model, test_data)
rf_rmse  <- sqrt(mean((actual - rf_pred)^2))
rf_mae   <- mean(abs(actual - rf_pred))
rf_r2    <- cor(actual, rf_pred)^2

# ============================================================
# STEP 16: HARE Algorithm
# ============================================================
features <- c("meals_served", "kitchen_staff", "past_waste_kg",
              "special_event", "temperature_C", "humidity_percent",
              "day_of_week", "staff_efficiency", "demand_pressure",
              "env_score", "event_risk", "waste_ratio",
              "is_weekend", "day", "month")

formula_fixed <- as.formula(
  paste("food_waste_kg ~", paste(features, collapse = " + "))
)

# Level 1: Base Linear Estimator
hare_lvl1   <- lm(formula_fixed, data = train_data)
train_resid <- train_data$food_waste_kg - predict(hare_lvl1, train_data)

# Level 2: Ghost Model with Asymmetric Weights
train_risk_weights <- ifelse(train_resid > 0, 10.0, 1.0)

hare_ghost <- randomForest(
  x       = train_data[, features],
  y       = train_resid,
  weights = train_risk_weights,
  ntree   = 500,
  importance = TRUE
)

# HARE Prediction Function
predict_hare <- function(newdata) {
  p1      <- predict(hare_lvl1,  newdata)
  p_ghost <- predict(hare_ghost, newdata[, features])
  return(p1 + p_ghost)
}

# Evaluate HARE
test_data$hare_pred <- predict_hare(test_data)
hare_rmse <- sqrt(mean((actual - test_data$hare_pred)^2))
hare_mae  <- mean(abs(actual - test_data$hare_pred))

cat("\n--- HARE Algorithm ---\n")
cat("RMSE:", round(hare_rmse, 4), "\n")
cat("MAE: ", round(hare_mae,  4), "\n")



# ============================================================
# STEP 17: Full Model Comparison Table
# ============================================================
results_final <- data.frame(
  Model = c("Baseline LM", "Random Forest", "HARE"),
  RMSE  = round(c(rmse,  rf_rmse,  hare_rmse), 4),
  MAE   = round(c(mae,   rf_mae,   hare_mae),  4),
  R2    = round(c(r2,    rf_r2,    cor(actual, test_data$hare_pred)^2), 4)
)

cat("\n--- Model Comparison ---\n")
print(results_final)

png("E:/r project/model_comparison.png", width = 800, height = 600)
barplot(
  height     = c(rmse, rf_rmse, hare_rmse),
  names.arg  = c("Baseline LM", "Random Forest", "HARE"),
  col        = c("blue", "orange", "green"),
  main       = "RMSE Comparison Across Models",
  ylab       = "RMSE",
  ylim       = c(0, max(rmse, rf_rmse, hare_rmse) * 1.2)
)
dev.off()

# ============================================================
# STEP 18: Penalty Sensitivity Analysis
# ============================================================
penalties        <- c(1, 3, 5, 7, 10)
results_penalty  <- data.frame()

for (p in penalties) {
  weights  <- ifelse(train_resid > 0, p, 1)
  temp_rf  <- randomForest(
    x       = train_data[, features],
    y       = train_resid,
    weights = weights,
    ntree   = 200
  )
  temp_pred      <- predict(temp_rf, test_data[, features])
  final_pred     <- predict(hare_lvl1, test_data) + temp_pred
  temp_shortfall <- sum((actual - final_pred) > 5)
  results_penalty <- rbind(results_penalty,
                           data.frame(Penalty = p,
                                      Shortfall = temp_shortfall))
}

cat("\n--- Penalty Sensitivity ---\n")
print(results_penalty)

png("E:/r project/penalty_sensitivity.png", width = 800, height = 600)
ggplot(results_penalty, aes(x = Penalty, y = Shortfall)) +
  geom_line(color = "darkred", linewidth = 1.2) +
  geom_point(color = "red", size = 4) +
  geom_vline(xintercept = 10, linetype = "dashed", color = "blue") +
  annotate("text", x = 10.2, y = max(results_penalty$Shortfall),
           label = "Chosen Penalty = 10", hjust = 0, color = "blue") +
  labs(title    = "Asymmetric Penalty vs Critical Shortfall Events",
       subtitle = "Optimal penalty selection for HARE ghost layer",
       x = "Under-prediction Penalty Multiplier",
       y = "Critical Shortfall Events (>5kg)") +
  theme_minimal(base_size = 14)
dev.off()

# ============================================================
# STEP 19: Operational Risk (Shortfall) Comparison
# ============================================================
threshold      <- 5
base_shortfall <- sum((actual - base_pred)          > threshold)
rf_shortfall   <- sum((actual - rf_pred)             > threshold)
hare_shortfall <- sum((actual - test_data$hare_pred) > threshold)

risk_results <- data.frame(
  Model               = c("Baseline LM", "Random Forest", "HARE"),
  Critical_Shortfalls = c(base_shortfall, rf_shortfall, hare_shortfall)
)

cat("\n--- Shortfall Risk Comparison ---\n")
print(risk_results)

png("E:/r project/shortfall_comparison.png", width = 800, height = 600)
barplot(
  height    = c(base_shortfall, rf_shortfall, hare_shortfall),
  names.arg = c("Baseline LM", "Random Forest", "HARE"),
  col       = c("blue", "orange", "green"),
  main      = "Critical Food Shortage Risk Comparison",
  ylab      = "Shortfall Events (>5kg underprediction)"
)
dev.off()

# Safety Table
base_res <- actual - base_pred
hare_res <- actual - test_data$hare_pred

safety_table <- data.frame(
  Metric   = c("Critical Shortfalls (>5kg)",
               "Max Shortfall Error (kg)",
               "Bias (Mean Error)"),
  Baseline = c(base_shortfall,
               round(max(base_res), 2),
               round(mean(base_res), 3)),
  HARE     = c(hare_shortfall,
               round(max(hare_res), 2),
               round(mean(hare_res), 3))
)

cat("\n--- Operational Risk Table ---\n")
print(safety_table)

# Shortfall Mass
shortfall_mass_base <- sum(base_res[base_res > 0], na.rm = TRUE)
shortfall_mass_hare <- sum(hare_res[hare_res > 0], na.rm = TRUE)
risk_mitigation_kg  <- shortfall_mass_base - shortfall_mass_hare
reduction_rate      <- (risk_mitigation_kg / shortfall_mass_base) * 100

cat("\n--- HARE Operational Impact ---\n")
cat("Baseline Shortfall Mass:", round(shortfall_mass_base, 2), "kg\n")
cat("HARE Shortfall Mass:    ", round(shortfall_mass_hare, 2), "kg\n")
cat("Risk Mitigated:         ", round(risk_mitigation_kg,  2), "kg\n")
cat("Reduction Rate:         ", round(reduction_rate, 2), "%\n")
cat("Meals Saved from Shortage:", round(risk_mitigation_kg / 0.4, 0), "\n")

# Residual Boxplot
plot_data <- data.frame(
  Residuals = c(base_res, hare_res),
  Model     = rep(c("Baseline", "HARE"), each = length(actual))
)

png("E:/r project/hare_vs_baseline_box.png", width = 800, height = 600)
ggplot(plot_data, aes(x = Model, y = Residuals, fill = Model)) +
  geom_boxplot() +
  geom_hline(yintercept = 0,  linetype = "dashed", color = "red") +
  geom_hline(yintercept = 5,  linetype = "dotted", color = "darkred") +
  annotate("text", x = 1.5, y = 7,
           label = "DANGER ZONE (Shortfall)", color = "darkred") +
  labs(title    = "Operational Risk Profile: HARE vs Baseline",
       subtitle = "HARE shifts distribution below the Danger Zone",
       y = "Residual Error (Actual - Predicted) kg") +
  theme_minimal()
dev.off()

# ============================================================
# STEP 20: HARE Cross-Validation 
# ============================================================
set.seed(123)
k            <- 5
folds        <- sample(rep(1:k, length.out = nrow(train_data)))
hare_cv_rmse <- c()
hare_cv_mae  <- c()

for (i in 1:k) {
  fold_train <- train_data[folds != i, ]
  fold_test  <- train_data[folds == i, ]
  
  fold_lvl1   <- lm(formula_fixed, data = fold_train)
  fold_resid  <- fold_train$food_waste_kg - predict(fold_lvl1, fold_train)
  fold_weights <- ifelse(fold_resid > 0, 10.0, 1.0)
  
  fold_ghost <- randomForest(
    x       = fold_train[, features],
    y       = fold_resid,
    weights = fold_weights,
    ntree   = 300
  )
  
  fold_final  <- predict(fold_lvl1, fold_test) +
    predict(fold_ghost, fold_test[, features])
  fold_actual <- fold_test$food_waste_kg
  
  hare_cv_rmse <- c(hare_cv_rmse, sqrt(mean((fold_actual - fold_final)^2)))
  hare_cv_mae  <- c(hare_cv_mae,  mean(abs(fold_actual - fold_final)))
}

cat("\n--- HARE 5-Fold Cross-Validation ---\n")
cat("Mean CV RMSE:", round(mean(hare_cv_rmse), 4), "\n")
cat("Mean CV MAE: ", round(mean(hare_cv_mae),  4), "\n")
cat("SD CV RMSE:  ", round(sd(hare_cv_rmse),   4), "\n")

# Baseline CV for comparison
train_control <- trainControl(method = "cv", number = 5)
cv_model <- train(
  food_waste_kg ~ .,
  data      = train_data,
  method    = "lm",
  trControl = train_control
)

cat("\n--- Baseline 5-Fold Cross-Validation ---\n")
print(cv_model$results[, c("RMSE", "Rsquared", "MAE")])

# ============================================================
# STEP 21: SHAP Explainability 
# ============================================================
set.seed(42)
bg_sample <- train_data[sample(nrow(train_data), 100), features]

s_values <- kernelshap(
  hare_ghost,
  X      = test_data[1:60, features],
  bg_X   = bg_sample,
  degree = 2
)

shap_matrix <- s_values$S
cat("Any NA in SHAP:", anyNA(shap_matrix), "\n")

shap_imp <- data.frame(
  Feature    = features,
  Importance = colMeans(abs(shap_matrix))
)
shap_imp <- shap_imp[order(shap_imp$Importance, decreasing = TRUE), ]

cat("\n--- SHAP Feature Importance ---\n")
print(shap_imp)

png("E:/r project/hare_shap_final.png", width = 1000, height = 800, res = 120)
ggplot(shap_imp, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", aes(fill = Importance)) +
  scale_fill_gradient(low = "#76c7c0", high = "#2a52be") +
  coord_flip() +
  labs(
    title    = "Feature Attribution: HARE Ghost Layer",
    subtitle = "Quantifying Feature Impact on Asymmetric Risk Correction",
    x        = "Predictors",
    y        = "Mean |SHAP Value|",
    caption  = "Calculated via Kernel SHAP | Framework: HARE"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title    = element_text(face = "bold", size = 16),
        legend.position = "none")
dev.off()

# ============================================================
# STEP 22: Shortage Risk Confusion Matrix
# ============================================================
high_waste_threshold <- median(actual)

actual_flag   <- ifelse(actual           > high_waste_threshold, 1, 0)
baseline_flag <- ifelse(predictions      > high_waste_threshold, 1, 0)
hare_flag     <- ifelse(test_data$hare_pred > high_waste_threshold, 1, 0)

cat("\n--- Baseline Shortage Detection ---\n")
print(confusionMatrix(factor(baseline_flag), factor(actual_flag), positive = "1"))

cat("\n--- HARE Shortage Detection ---\n")
print(confusionMatrix(factor(hare_flag), factor(actual_flag), positive = "1"))

cat("\nAll steps complete. Check E:/r project/ for all saved plots.\n")
