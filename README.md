# Hull Tactical — Market Prediction

## Overview
This repository contains a time-series machine learning pipeline for market direction classification using the Hull Tactical dataset.  
It implements an end-to-end workflow: preprocessing, feature engineering, model tuning, ensemble learning, evaluation, threshold-based decision rules, backtesting, and explainability.

## Project Structure
- `notebook/`
  - `main.ipynb`: main experimentation and results notebook
- `kit/`
  - custom utilities for data loading, preprocessing, feature engineering, scaling, plotting, backtesting, and explainability
- `Data/`
  - training and test files used in the notebook
- `requirements.txt`
  - Python dependencies

## Method Summary
1. **Preprocessing**
   - Log transformation
   - Missing value handling
2. **Feature Engineering**
   - Derived features built from `forward_returns`
3. **Target Construction**
   - 3-class labeling via `categorize_y(..., bin=0.003)`
4. **Modeling**
   - CatBoost
   - SVM (RBF)
   - Gradient Boosting
5. **Hyperparameter Optimization**
   - Optuna with `TimeSeriesSplit`
6. **Ensemble**
   - Soft voting classifier combining the three models
7. **Decision Calibration**
   - Custom probability threshold for the “Drop” class to control risk
8. **Evaluation**
   - Holdout test metrics + time-series cross-validation
9. **Interpretability**
   - Permutation importance
   - SHAP for CatBoost
  
   
