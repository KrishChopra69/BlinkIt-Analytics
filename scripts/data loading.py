import pandas as pd

# Load the dataset
file_path = "C:/Users/91931/Desktop/BlinkIt Project/BlinkIT Grocery Data.xlsx"
df = pd.read_excel(file_path)

# Display basic info
print(df.info())
print(df.head())

# Transform categories
df['Item Fat Content'] = df['Item Fat Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

# Filter dataset to keep only "Low Fat" and "Regular"
df = df[df['Item Fat Content'].isin(['Low Fat', 'Regular'])]

# Save cleaned data
df.to_csv("C:/Users/91931/Desktop/BlinkIt Project/data/cleaned_data.csv", index=False)