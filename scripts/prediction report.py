import pandas as pd
import joblib

base_dir = r"C:\Users\91931\Desktop\BlinkIt Project"
featured_data_path = f"{base_dir}/data/featured_data.csv"
df = pd.read_csv(featured_data_path)

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

X = pd.get_dummies(df[categorical_cols], drop_first=True)
important_cols = [col for col in X.columns if 'Item Type_' not in col or col in [
    'Item Type_Canned', 'Item Type_Dairy', 'Item Type_Frozen Foods', 
    'Item Type_Household', 'Item Type_Snack Foods']]
X = pd.concat([X[important_cols], df[numerical_cols]], axis=1)

for col in required_cols:
    if col not in X.columns:
        X[col] = 0

for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())

model_path = f"{base_dir}/models/random_forest_final.pkl"
model = joblib.load(model_path)

df['Predicted_Sales'] = model.predict(X[required_cols])

# Keep detailed report
report_cols = ['Item Identifier', 'Outlet Identifier', 'Outlet Location Type', 
               'Outlet Size', 'Item Type', 'Item Fat Content', 'Sales', 
               'Predicted_Sales']
report_df = df[report_cols].copy()
report_df['Sales_Difference'] = report_df['Sales'] - report_df['Predicted_Sales']

report_path = f"{base_dir}/data/sales_prediction_report_detailed.csv"
report_df.to_csv(report_path, index=False)
print(f"Detailed sales prediction report saved to {report_path}, Rows: {len(report_df)}")