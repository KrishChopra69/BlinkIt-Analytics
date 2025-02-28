import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load featured data
featured_data_path = os.path.join(base_dir, '../data/featured_data.csv')
df = pd.read_csv(featured_data_path)

# Prepare features and target
categorical_cols = ['Item Fat Content', 'Item Type', 'Outlet Location Type', 
                    'Outlet Size', 'Outlet Type']
numerical_cols = ['Item Weight', 'Item Visibility', 'Visibility_per_Weight', 
                  'Rating', 'Visibility_x_ItemType', 'Weight_x_ItemType']

# One-hot encoding with filtering
X = pd.get_dummies(df[categorical_cols], drop_first=True)
important_cols = [col for col in X.columns if 'Item Type_' not in col or col in [
    'Item Type_Canned', 'Item Type_Dairy', 'Item Type_Frozen Foods', 
    'Item Type_Household', 'Item Type_Snack Foods']]
X = pd.concat([X[important_cols], df[numerical_cols]], axis=1)
y = df['Sales']

# Handle missing values
X.loc[:, numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=35, max_features=0.9, min_samples_split=2, 
    min_samples_leaf=1, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print("CV R² Scores:", cv_scores)
print("Mean CV R²:", cv_scores.mean())

# Save the model
models_dir = os.path.join(base_dir, '../models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_final.pkl'))
print("Model saved as 'random_forest_final.pkl'.")