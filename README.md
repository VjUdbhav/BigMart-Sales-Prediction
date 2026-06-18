# BigMart Sales Prediction

Predicting `Item_Outlet_Sales` for BigMart retail outlets using an XGBoost regression pipeline, built from raw product/outlet attributes through to a tuned, leakage-free model.

## Project Overview

BigMart operates a chain of retail outlets across multiple cities. This project builds a regression model that predicts the sales of a given product at a given outlet, using attributes of the product (weight, price, type, fat content) and the outlet (size, location tier, format, age). The end-to-end pipeline covers data inspection, exploratory data analysis, cleaning, feature engineering, encoding, hyperparameter-tuned XGBoost training, evaluation, and model persistence.

## Problem Statement

Given historical sales data for 1,559 products across 10 outlets, predict `Item_Outlet_Sales` — the sales of each product at each outlet — for a held-out set of product/outlet combinations. Accurate predictions support inventory planning, outlet-level forecasting, and pricing strategy decisions.

## Dataset Description

| File | Rows | Columns | Description |
|---|---|---|---|
| `Train.csv` | 8,523 | 12 | Labeled training data, includes target |
| `Test.csv` | 5,681 | 11 | Unlabeled data for final predictions |

**Features:**

| Column | Description |
|---|---|
| `Item_Identifier` | Unique product ID |
| `Item_Weight` | Weight of the product |
| `Item_Fat_Content` | Whether the product is low fat or regular |
| `Item_Visibility` | % of total display area allocated to this product in the outlet |
| `Item_Type` | Product category |
| `Item_MRP` | Maximum Retail Price (listed price) |
| `Outlet_Identifier` | Unique outlet ID |
| `Outlet_Establishment_Year` | Year the outlet was established |
| `Outlet_Size` | Outlet size (Small / Medium / High) |
| `Outlet_Location_Type` | Tier of the city the outlet is in |
| `Outlet_Type` | Outlet format (Grocery Store / Supermarket Type1–3) |
| `Item_Outlet_Sales` | **Target** — sales of the product at that outlet |

**Known data quality issues, addressed in the notebook:** `Item_Weight` (2,439 missing across train+test) and `Outlet_Size` (4,016 missing) both have missing values; `Item_Fat_Content` has inconsistent category labels (`LF`, `low fat`, `reg`); `Item_Visibility` contains a meaningful number of zero values that are not physically realistic.

## Technologies Used

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — preprocessing, train/test split, `RandomizedSearchCV`
- **XGBoost** — gradient boosted regression model
- **Joblib** — model persistence
- **Jupyter / Google Colab** — notebook environment

## Installation Instructions

```bash
# Clone the repository
https://github.com/VjUdbhav/BigMart-Sales-Prediction.git
cd BigMart-Sales-Prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook BigMart_Sales_Prediction.ipynb
```

To run in **Google Colab** instead: upload `BigMart_Sales_Prediction.ipynb`, then upload `data/Train.csv` and `data/Test.csv` to the Colab session (or mount Google Drive — an optional cell for this is included near the top of the notebook).

## Project Workflow

1. **Load & Inspect** — shapes, dtypes, missing values, summary statistics, unique category counts
2. **Exploratory Data Analysis** — 11+ visualizations covering distributions, relationships, and outliers
3. **Data Cleaning** — category standardization, missing value imputation, duplicate checks
4. **Feature Engineering** — `Outlet_Age`, `Item_Category`, `Item_Visibility_MeanRatio`, `Item_MRP_Per_Weight`, `Item_MRP_Bin`
5. **Encoding** — ordinal encoding for ordered categories, one-hot for nominal categories, label encoding for outlet identity
6. **Train/Validation Split** — 80/20 split on labeled data with `random_state=42`
7. **Model Training** — XGBoost Regressor tuned via `RandomizedSearchCV` (5-fold CV, 60 candidate configurations) with early stopping
8. **Evaluation** — R², MAE, RMSE on train and validation sets
9. **Feature Importance** — ranked table and bar chart
10. **Prediction & Persistence** — predictions on `Test.csv`, model saved with `joblib`

## EDA Findings

- `Item_Outlet_Sales` is right-skewed with a long tail of high-value sales, typical of retail revenue data.
- `Item_MRP` shows four visually distinct price bands, each associated with a step-wise jump in average sales — the single strongest pattern in the raw data, and the motivation for the engineered `Item_MRP_Bin` feature.
- `Outlet_Type` strongly separates sales levels: Grocery Stores sell substantially less than any Supermarket format, while `Supermarket Type3` shows the highest median sales.
- Roughly 6% of rows have `Item_Visibility == 0`, which is not physically realistic (every shelved item has some visibility) — these are treated as missing and imputed.
- `Item_Fat_Content` originally contained 5 labels (`Low Fat`, `LF`, `low fat`, `Regular`, `reg`) collapsing to 2 true categories, and was meaningless for non-consumable products (household, hygiene items were all mislabeled `Low Fat`).
- Missing `Outlet_Size` values are concentrated entirely in 3 specific outlets (not random row-level gaps), which informed an outlet-type-based imputation strategy rather than a single global fill value.

## Model Architecture

**Algorithm:** XGBoost Regressor (`reg:squarederror` objective)

**Hyperparameter tuning:** `RandomizedSearchCV`, 5-fold cross-validation, 60 sampled configurations, scored on negative RMSE, searching over:

- `n_estimators`, `learning_rate`, `max_depth`
- `subsample`, `colsample_bytree`, `min_child_weight`
- `reg_alpha`, `reg_lambda` (L1/L2 regularization)

**Best configuration found:**

| Parameter | Value |
|---|---|
| `n_estimators` | 800 |
| `learning_rate` | 0.01 |
| `max_depth` | 3 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.9 |
| `min_child_weight` | 15 |
| `reg_alpha` | 5 |
| `reg_lambda` | 10 |

The final model is refit with these parameters using early stopping (50 rounds, monitored on the validation set) to select the optimal number of boosting rounds without manual guesswork.

## Evaluation Metrics

| Metric | Train | Validation |
|---|---|---|
| R² Score | 0.6106 | 0.6109 |
| MAE | 759.03 | 725.04 |
| RMSE | 1073.26 | 1028.41 |

Train and validation scores are nearly identical (within 0.0003 R²), which indicates the model generalizes well and is not overfitting. This is the expected behavior of a correctly regularized model with no data leakage: all imputation and feature engineering used only static product/outlet attributes, never the target variable.

An R² in this range is consistent with widely reported results on this dataset — `Item_Outlet_Sales` contains substantial inherent variance (promotions, regional demand, and other unobserved factors) that no feature set built from these columns alone can fully explain. This reflects a realistic, honestly-validated ceiling rather than an inflated metric.

## Results

**Top predictive features**, by XGBoost importance:

1. `Item_MRP_Bin` (engineered price tier)
2. `Item_MRP` (raw price)
3. `Item_Visibility_MeanRatio` (engineered, relative shelf visibility)
4. `Outlet_Type` (Supermarket Type3, Type1)
5. `Outlet_Identifier`, `Outlet_Age`, `Outlet_Size`

Pricing dominates the model's decisions, followed by outlet format and relative product visibility. Individual `Item_Type` categories contribute comparatively little on their own.

See `outputs/feature_importance.png` for the full chart and `outputs/predictions.csv` for the generated test-set predictions.

## Business Insights

1. **Pricing tier matters more than product category.** Outlets may gain more from price-tier strategy than from category-level stocking decisions.
2. **Outlet format is a structural driver of revenue.** Grocery Stores underperform Supermarkets regardless of assortment, suggesting format-level investment decisions (e.g., upgrading store type) could matter more than micro-level merchandising changes.
3. **Shelf visibility has a real, secondary effect.** Items displayed more prominently than typical for that product tend to sell differently, supporting continued investment in in-store merchandising.
4. **Significant unexplained variance is expected.** The dataset is a static snapshot without promotional or seasonal data, so ~39% of variance likely reflects factors outside what's captured here — a natural target for future data collection.

## Future Improvements

- Incorporate promotional/seasonal data if it becomes available — the current dataset is a single static snapshot.
- Explore target encoding for `Outlet_Identifier` and `Item_Type` with proper nested cross-validation to avoid leakage.
- Compare against LightGBM, CatBoost, and a regularized linear baseline; consider stacking.
- Apply SHAP values for richer, per-prediction interpretability beyond global feature importance.
- Experiment with a log-transformed target to assess whether it improves performance on the right-skewed sales distribution.


## Repository Structure

```
BigMart-Sales-Prediction/
│
├── BigMart_Sales_Prediction.ipynb   # Full end-to-end notebook
├── data/
│   ├── Train.csv
│   └── Test.csv
├── models/
│   └── xgboost_model.pkl            # Persisted final model
├── outputs/
│   ├── predictions.csv              # Test set predictions
│   └── feature_importance.png       # Feature importance chart
├── requirements.txt
└── README.md
```

## Author

Udbhav Vijay

---

*This project was built and validated against the BigMart Sales dataset (Train.csv: 8,523 rows; Test.csv: 5,681 rows). All metrics reported above were produced by executing the notebook end-to-end.*
