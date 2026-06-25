# PM25-Prediction-Code

Source code for PM2.5 prediction models in arid regions.

This repository contains the implementation code for the paper: 
**"Prediction of PM2.5 Concentrations in Arid Regions Using a Three-Level LSTM-CatBoost Hybrid Model"**.

---

## Requirements

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

---

## Repository Structure

| Script | Description | Figure in Paper |
|--------|-------------|-----------------|
| `catboost.py` | CatBoost model training. Trains a gradient boosting decision tree model using pollutant features (NO₂, SO₂, CO, O₃) to predict PM2.5 concentrations. Evaluates performance with R², RMSE, and RPD metrics on both calibration and validation sets. | - |
| `lstm.py` | LSTM model training. Trains a Long Short-Term Memory neural network using time-series meteorological features (T_max, T_min, PRE, WS_max, DEWP, SP, RH) to capture temporal dependencies in PM2.5 evolution. | - |
| `knn.py` | KNN model. Implements K-Nearest Neighbors regression for PM2.5 prediction baseline comparison. | - |
| `lightgbm.py` | LightGBM model. Trains a leaf-wise gradient boosting model for PM2.5 prediction baseline comparison. | - |
| `mlp.py` | MLP model. Trains a Multi-Layer Perceptron neural network with multiple hidden layers for PM2.5 prediction baseline comparison. | - |
| `randomforest.py` | Random Forest model. Trains an ensemble of decision trees using bagging strategy for PM2.5 prediction baseline comparison. | - |
| `svm.py` | SVM model. Implements Support Vector Regression with kernel-based nonlinear mapping for PM2.5 prediction baseline comparison. | - |
| `xgboost.py` | XGBoost model. Trains an extreme gradient boosting model with regularization for PM2.5 prediction baseline comparison. | - |
| `taylor_diagram.py` | Taylor diagram visualization. Generates a Taylor diagram comparing all eight single models (CatBoost, LSTM, RF, XGBoost, SVM, MLP, KNN, LightGBM) by correlation coefficient, standard deviation, and RMSE on both calibration and validation sets. | Figure 6 |
| `stacking_three_level_fusion.py` | Three-level stacking fusion. **Core model of this study.** Implements a three-level ensemble: Level 1 runs CatBoost (pollutants) and LSTM (meteorology) in parallel; Level 2 concatenates predictions and feeds to a CatBoost meta-learner for bias correction; Level 3 performs final adaptive fusion. Achieves best performance: R²=0.93, RMSE=12.00 μg/m³, RPD=3.77. | Figure 5a |
| `dynamic_gated_fusion.py` | Dynamic gated fusion. Alternative fusion method that learns adaptive weights to dynamically combine CatBoost and LSTM predictions based on gating mechanism. | Figure 5b |
| `feature_concatenation_fusion.py` | Feature concatenation fusion. Alternative fusion method that directly concatenates all raw features before single-model training. | Figure 5c |
| `catboost_shap.py` | CatBoost SHAP analysis. Computes SHAP values for the CatBoost model to interpret feature contributions. Identifies CO as the dominant pollutant driver and quantifies positive/negative effects of NO₂, SO₂, and O₃. | Figure 8a |
| `lstm_shap.py` | LSTM SHAP analysis. Computes SHAP values for the LSTM model to interpret temporal meteorological feature contributions. Identifies T_min (daily minimum temperature) as the key meteorological driver and T_max as negatively correlated with PM2.5. | Figure 8b |
| `stacking_overall_shap.py` | Stacking overall SHAP. Computes SHAP values for the final three-level stacking fusion model, combining both pollutant and meteorological features. Reveals CO, SO₂, and T_min as the top three overall drivers. | Figure 8c |
| `pm25_raincloud.py` | Raincloud plot. Generates raincloud plots showing PM2.5 distribution characteristics for calibration set, validation set, and full dataset, combining box plots, violin plots, and scatter points. | Figure 4 |
| `time_scale_comparison.py` | Time scale comparison. Generates four-panel figure showing PM2.5 variations at daily, monthly, seasonal, and interannual scales (2020-2024), revealing "high in winter, low in summer" pattern and 20% annual decrease trend. | Figure 9 |
| `pm25_scenario_trends.py` | Scenario simulation trends. Simulates PM2.5 evolution under four scenarios (S₀: baseline; S₁: +0.81°C warming; S₂: 20% CO reduction; S₃: combined) for January-March 2050, with 75 μg/m³ pollution alert threshold. | Figure 10 |

---

## Quick Test

```bash
python tests/test_imports.py
