import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding Data")

file_name = 'sales_data2.csv'
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found")
    exit()

df = pd.read_csv(file_name)
print("Successfully Loaded")
print(f"Shape of the Dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print(df.head())
print(df.info())
print(df.describe())

print("Handling missing values:")
print(df.isnull().sum())

# With using median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"Median Age: {median_age}")

median_spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)
print(f"Median Spending: {median_spending}")

# Distribution Analysis
plt.figure(figsize=(7, 4))
df['Spending'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Spending Amount")
plt.xlabel("Spending")
plt.ylabel("Number of Customers")
plt.show()

# Correlation Matrix
correlation = df.corr(numeric_only=True)
print(correlation)

print("Plotting Correlation Matrix")
plt.figure(figsize=(7, 4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("Finding Outliers in Age")
outliers = df[df['Age'] > 100]
print(f"Found Outliers ({len(outliers)}):")
print(outliers)
