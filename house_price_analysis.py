import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load the cleaned sales dataset
df = pd.read_csv('sale_clean.csv')

# Data Cleaning
# ==============================================

# Clean 'prezzo' column
if df['prezzo'].dtype == 'object':
    df['prezzo'] = df['prezzo'].str.replace('€', '', regex=False)
    df['prezzo'] = df['prezzo'].str.replace('.', '', regex=False)
    df['prezzo'] = df['prezzo'].str.replace(',', '.', regex=False)
    df['prezzo'] = pd.to_numeric(df['prezzo'], errors='coerce')

# Ensure type is float
df['prezzo'] = df['prezzo'].astype(float)

# Drop rows with invalid price values
df = df[df['prezzo'].notna()]
df = df[(df['prezzo'] > 1000) & (df['prezzo'] < 5_000_000)]  # Reasonable price range

# Drop columns with too many missing values or not useful for prediction
df = df.drop(columns=['datetime', 'quartiere'], errors='ignore')

# Handle missing values - we'll do this in the pipeline later

# EDA - Exploratory Data Analysis
# ==============================================

# Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df['prezzo']), kde=True, bins=40)  # Log transform for better visualization
plt.title('Log Distribution of House Prices')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('price_distribution.png')
plt.show()
plt.close()

# Correlation heatmap (numerical features)
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()

# Feature Engineering
# ==============================================

# Separate features and target
y = df['prezzo']
X = df.drop(columns=['prezzo'])

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing Pipeline
# ==============================================

# Create transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

# Split data before any transformation to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# ==============================================

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices (€)')
    plt.ylabel('Predicted Prices (€)')
    plt.title(f'Actual vs Predicted Prices ({model.__class__.__name__})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{model.__class__.__name__}.png')
    plt.show()
    plt.close()

    return pipeline, rmse, r2

# Linear Regression
lr_model = LinearRegression()
lr_pipeline, rmse_lr, r2_lr = evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)  # Limited depth to prevent overfitting
dt_pipeline, rmse_dt, r2_dt = evaluate_model(dt_model, X_train, y_train, X_test, y_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)  # Limited depth
rf_pipeline, rmse_rf, r2_rf = evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# Save the best model (Random Forest in this case)
joblib.dump(rf_pipeline, 'random_forest_pipeline.pkl')
print("Model pipeline saved successfully.")