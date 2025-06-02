# Santander Customer Transaction Prediction: Gradient Boosting Comparison

This project compares the performance of two popular gradient boosting frameworks — **LightGBM** and **CatBoost** — using the [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) dataset. We explore both raw and manually engineered features to understand how different modeling approaches and feature strategies affect predictive performance.

---

## Project Highlights

-  **Dataset**: Santander Customer Transaction Prediction (from Kaggle)
-  **Models**: LightGBM, CatBoost
-  **Feature Engineering**: Manual statistical features (mean, std, kurtosis, etc.)
-  **Goal**: Compare raw vs. feature-engineered inputs using ROC AUC score

---

## File Structure

- `Santander_GBM_LightGBM_CatBoost_FeatureEngineering.ipynb`: Jupyter notebook with code, models, results
- `README.md`: This file

---

## Experiments

We trained each model under two setups:

1. **Raw Features**: Using the dataset without manual feature engineering
2. **Manual Features**: Adding handcrafted statistical features

Each configuration was evaluated using **ROC AUC** on a validation split.

| Rank | Model           | ROC AUC   |
|------|------------------|-----------|
| 1 | **CatBoost (Raw)**  | **0.894776** |
| 2 | CatBoost (Manual)   | 0.894034 |
| 3 | LightGBM (Raw)      | 0.888040 |
| 4 | LightGBM (Manual)    | 0.887241 |

---

## Key Learnings

- **CatBoost performs best out-of-the-box**, handling high-dimensional raw features very well.
- **Manual feature engineering benefits LightGBM** slightly, but not CatBoost.
- Feature engineering still holds value, especially when model flexibility is limited.
- Gradient boosting remains a powerful method for tabular data.

---

## Requirements

- Python 3.7+
- Libraries:
  - `lightgbm`
  - `catboost`
  - `scikit-learn`
  - `pandas`, `numpy`, `matplotlib`, `seaborn`

Install via pip:
```bash
pip install lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```

Dataset Source
Kaggle: Santander Customer Transaction Prediction

You must agree to the competition rules to download the dataset.

## Future Improvements

-Hyperparameter tuning (e.g., with Optuna or GridSearchCV)
-Advanced feature generation (e.g., interactions, autoencoders)
-Model stacking and ensembling
-Feature importance visualization with SHAP

