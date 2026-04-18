# HARE: Risk-Aware Food Waste Prediction

## 📖 Overview
This project proposes a novel machine learning framework called **HARE (Hierarchical Asymmetric Residual Expansion)** for predicting food waste in industrial catering systems.

## 🚀 Key Idea
Unlike traditional models, HARE prioritizes **underprediction risk (food shortage)** using asymmetric learning.

## 🧠 Model Architecture
- Level 1: Linear Regression (baseline)
- Level 2: Random Forest (residual correction)
- Asymmetric penalty for underprediction

## 📊 Features
- meals_served
- kitchen_staff
- past_waste_kg
- environmental factors
- engineered features (demand_pressure, env_score)

## 📈 Results
- Comparable RMSE to baseline
- Significant reduction in **critical shortfalls**
- Improved operational safety

## 🛠 Tech Stack
- R
- Random Forest
- Kernel SHAP
- ggplot2

## 📌 Future Work
- Real-time deployment
- Integration with IoT systems
