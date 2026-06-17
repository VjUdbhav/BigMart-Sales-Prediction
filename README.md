# BigMart-Sales-Prediction


Predicting `Item_Outlet_Sales` for BigMart outlets using an XGBoost regressor.

## Project Overview
An end-to-end regression project covering EDA, leakage-safe preprocessing,
feature engineering, hyperparameter tuning, and model persistence.

## Problem Statement
Given product and outlet attributes, predict the sales of each product at a
particular store to help with inventory and demand planning.

## Dataset Description
- **Train.csv** — 8,523 rows (includes target `Item_Outlet_Sales`)
- **Test.csv** — 5,681 rows (no target)

| Feature | Description |
|---|---|
| Item_Identifier | Unique product ID |
| Item_Weight | Product weight |
| Item_Fat_Content | Fat content category |
| Item_Visibility | Shelf visibility share |
| Item_Type | Product category |
| Item_MRP | Maximum retail price |
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | Year store opened |
| Outlet_Size | Store size |
| Outlet_Location_Type | City tier |
| Outlet_Type | Store format |

Missing values: `Item_Weight`, `Outlet_Size`, and disguised zeros in `Item_Visibility`.

## Technologies Used
Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost, Joblib.
