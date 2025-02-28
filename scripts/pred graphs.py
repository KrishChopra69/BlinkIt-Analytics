import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Define the base directory
base_dir = r"C:\Users\91931\Desktop\BlinkIt Project"  # Update if needed

# Create graphs directory if it doesn't exist
graphs_dir = f"{base_dir}/graphs"
os.makedirs(graphs_dir, exist_ok=True)

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

for col in required_cols:
    if col not in X.columns:
        X[col] = 0

for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())

# Load the model
model_path = f"{base_dir}/models/random_forest_final.pkl"
model = joblib.load(model_path)

# Predict
df['Predicted_Sales'] = model.predict(X[required_cols])
df['Sales_Difference'] = df['Sales'] - df['Predicted_Sales']

# Graph 1: Outlet-Level Predicted vs. Actual Sales
plt.figure(figsize=(12, 6))
outlet_summary = df.groupby('Outlet Identifier')[['Sales', 'Predicted_Sales']].sum().reset_index()
outlet_summary.plot(kind='bar', x='Outlet Identifier', stacked=False, ax=plt.gca())
plt.title('Actual vs. Predicted Sales by Outlet')
plt.xlabel('Outlet Identifier')
plt.ylabel('Total Sales ($)')
plt.legend(['Actual Sales', 'Predicted Sales'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{graphs_dir}/outlet_sales_comparison.png")
plt.close()

# Graph 2: Sales Difference by Outlet Size
plt.figure(figsize=(10, 6))
sns.barplot(x='Outlet Size', y='Sales_Difference', data=df, estimator=sum, ci=None)
plt.title('Sales Difference (Actual - Predicted) by Outlet Size')
plt.xlabel('Outlet Size')
plt.ylabel('Total Sales Difference ($)')
plt.tight_layout()
plt.savefig(f"{graphs_dir}/sales_difference_by_size.png")
plt.close()

# Graph 3: Predicted Sales by Tier
plt.figure(figsize=(10, 6))
sns.barplot(x='Outlet Location Type', y='Predicted_Sales', data=df, estimator=sum, ci=None)
plt.title('Predicted Sales by City Tier')
plt.xlabel('City Tier')
plt.ylabel('Total Predicted Sales ($)')
plt.tight_layout()
plt.savefig(f"{graphs_dir}/predicted_sales_by_tier.png")
plt.close()

# Graph 4: Item Type Contribution
plt.figure(figsize=(12, 6))
item_summary = df.groupby('Item Type')['Predicted_Sales'].sum().reset_index()
item_summary = item_summary.sort_values('Predicted_Sales', ascending=False)
sns.barplot(x='Item Type', y='Predicted_Sales', data=item_summary)
plt.title('Predicted Sales by Item Type')
plt.xlabel('Item Type')
plt.ylabel('Total Predicted Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{graphs_dir}/predicted_sales_by_item_type.png")
plt.close()

# Graph 5: Fat Content Impact
plt.figure(figsize=(8, 6))
sns.barplot(x='Item Fat Content', y='Predicted_Sales', data=df, estimator=sum, ci=None)
plt.title('Predicted Sales by Fat Content')
plt.xlabel('Fat Content')
plt.ylabel('Total Predicted Sales ($)')
plt.tight_layout()
plt.savefig(f"{graphs_dir}/predicted_sales_by_fat_content.png")
plt.close()

print(f"Graphs saved in {graphs_dir}/")