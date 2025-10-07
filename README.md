# House Price Analysis & Prediction — Italy

This project demonstrates an end-to-end machine learning workflow for predicting housing prices in Italy: data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and saving a trained pipeline.

## Files

- `sale_clean.csv` — cleaned dataset used for modeling
- `house_price_analysis.py` — main analysis and model training script
- `random_forest_pipeline.pkl` — saved sklearn pipeline (best model)
- `Plots/` — visualizations (price distribution, correlation heatmap, predicted vs actual plots)

## Dataset

Original dataset source: https://www.kaggle.com/datasets/tommasoramella/italy-house-prices

The included `sale_clean.csv` is a preprocessed subset used for training and evaluation.

## Requirements

Python 3.8+

Recommended packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install quickly with:

pip install pandas numpy scikit-learn matplotlib seaborn joblib

## How to run

1. (Optional) Create and activate a virtual environment:

   python -m venv .venv
   .venv\\Scripts\\activate   # Windows (cmd.exe)

2. Install dependencies (see above).

3. To run the main training script and reproduce results:

   python house_price_analysis.py

4. To load the saved Random Forest pipeline and make predictions from Python:

```python
import joblib
import pandas as pd

pipe = joblib.load('random_forest_pipeline.pkl')
# example: predict on a small sample from the CSV
sample = pd.read_csv('sale_clean.csv').head(5)
preds = pipe.predict(sample)
print(preds)
```

## Results summary

- Models trained: Linear Regression, Decision Tree, Random Forest
- Best model: Random Forest (saved as `random_forest_pipeline.pkl`)
- Evaluation: RMSE and R² scores are shown in the script output and plots in `Plots/`.

## Next steps / ideas

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Model explainability with SHAP or permutation importance
- Deploy the model behind a small Flask API for inference

---

Author: Mohammadamin (Amin) Jahanimajd

University of Messina — BSc Data Analysis
