import pandas as pd
import numpy as np
import os

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load cleaned data
cleaned_data_path = os.path.join(base_dir, '../data/cleaned_data.csv')
df = pd.read_csv(cleaned_data_path)

# Feature Engineering
df['Avg_Sales_per_Outlet'] = df.groupby('Outlet Identifier')['Sales'].transform('mean')
df['Years_Since_Establishment'] = 2025 - df['Outlet Establishment Year']
df['Visibility_per_Weight'] = df['Item Visibility'] / df['Item Weight'].replace(0, 1)
df['Visibility_per_Weight'] = df['Visibility_per_Weight'].replace([np.inf, -np.inf], 0)
df['Sales_per_Outlet_Type'] = df.groupby('Outlet Type')['Sales'].transform('mean')
df['Visibility_x_ItemType'] = df['Item Visibility'] * df['Item Type'].astype(str).map(
    lambda x: pd.Series(df['Item Type']).value_counts()[x] / len(df)
)
df['Weight_x_ItemType'] = df['Item Weight'] * df['Item Type'].astype(str).map(
    lambda x: pd.Series(df['Item Type']).value_counts()[x] / len(df)
)
df['Rating_x_Visibility'] = df['Rating'] * df['Item Visibility']

# Cap outliers in Sales
Q1, Q3 = df['Sales'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df['Sales'] = df['Sales'].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# Save updated dataset
featured_data_path = os.path.join(base_dir, '../data/featured_data.csv')
df.to_csv(featured_data_path, index=False)
print("Feature engineering completed. Updated dataset saved as 'featured_data.csv'.")