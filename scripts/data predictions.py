import pandas as pd
import joblib

# Define the base directory
base_dir = "C:/Users/91931/Desktop/BlinkIt Project"

# Load featured data
featured_data_path = f"{base_dir}/data/featured_data.csv"
df = pd.read_csv(featured_data_path)

# Define required columns for prediction (19 total)
required_cols = [
    'Item Fat Content_Regular', 'Item Type_Canned', 'Item Type_Dairy',
    'Item Type_Frozen Foods', 'Item Type_Household', 'Item Type_Snack Foods',
    'Outlet Location Type_Tier 2', 'Outlet Location Type_Tier 3',
    'Outlet Size_Medium', 'Outlet Size_Small', 'Outlet Type_Supermarket Type1',
    'Outlet Type_Supermarket Type2', 'Outlet Type_Supermarket Type3',
    'Item Weight', 'Item Visibility', 'Visibility_per_Weight', 'Rating',
    'Visibility_x_ItemType', 'Weight_x_ItemType'
]

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

# Ensure all required columns exist, fill missing with 0 or median
for col in required_cols:
    if col not in X.columns:
        X[col] = 0

for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())

# Load the trained model
model_path = f"{base_dir}/models/random_forest_final.pkl"
model = joblib.load(model_path)

# Predict
df['Predicted_Sales'] = model.predict(X[required_cols])

# Keep only identifier(s) and Predicted_Sales
key_cols = ['Item Identifier', 'Outlet Identifier']  # Adjust based on your .xlsx
output_df = df[key_cols + ['Predicted_Sales']]

# Save predictions
output_path = f"{base_dir}/data/predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")