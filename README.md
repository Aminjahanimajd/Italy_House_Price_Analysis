# 🏠 House Price Analysis and Prediction in Italy

This project presents a full machine learning workflow to analyze and predict housing prices in Italy using a cleaned real estate dataset. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment of the best model.

## 📂 Dataset

The dataset is sourced from Kaggle:  
[Italy House Prices – Kaggle Dataset](https://www.kaggle.com/datasets/tommasoramella/italy-house-prices?resource=download)

The cleaned version of the dataset used for modeling is named: `sale_clean.csv`.

## 🧹 Data Cleaning
- Converted 'prezzo' (price) column to numeric format.
- Removed outliers with unrealistic price values.
- Dropped irrelevant or sparse columns such as `datetime` and `quartiere`.

## 📊 Exploratory Data Analysis (EDA)

Key visualizations were generated to understand the dataset:

- **Price Distribution** (log-transformed)
- **Correlation Heatmap** (numerical features)

Plots are saved under the `plots/` directory for reference.

| Visualization | File Path |
|---------------|-----------|
| Log Price Distribution | `plots/price_distribution.png` |
| Correlation Heatmap    | `plots/correlation_heatmap.png` |

---

## 🏗️ Feature Engineering
- Identified numerical and categorical columns.
- Used pipelines with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
- Applied `ColumnTransformer` for combined preprocessing.

## 🤖 Model Training & Evaluation

Three regression models were trained and compared:

| Model                   | RMSE (€)   | R² Score |
|------------------------|------------|----------|
| Linear Regression       | 374,218.64 | 0.4124   |
| Decision Tree Regressor | 385,286.93 | 0.3771   |
| Random Forest Regressor | 336,404.69 | 0.5251   |

Evaluation plots showing predicted vs actual prices are stored under `plots/`:

| Model                  | Plot Path |
|------------------------|-----------|
| Linear Regression       | `plots/actual_vs_predicted_LinearRegression.png` |
| Decision Tree Regressor | `plots/actual_vs_predicted_DecisionTreeRegressor.png` |
| Random Forest Regressor | `plots/actual_vs_predicted_RandomForestRegressor.png` |

---
## 🏆 Best Model
The **Random Forest Regressor** performed the best and was saved as a pipeline for future use:

joblib.dump(rf_pipeline, 'random_forest_pipeline.pkl')

## 📁 Project Structure
italy-house-price-analysis/
│
├── sale_clean.csv
├── random_forest_pipeline.pkl
├── house_price_analysis.py
├── plots/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted_LinearRegression.png
│   ├── actual_vs_predicted_DecisionTreeRegressor.png
│   └── actual_vs_predicted_RandomForestRegressor.png
└── README.md


## 📚 Requirements
Install required packages using:

bash pip install pandas numpy matplotlib seaborn scikit-learn joblib

## 👤 Author
Amin
Second-year Data Analysis student
University of Messina, Italy
Aspiring Machine Learning Engineer
