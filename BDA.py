from numpy.lib._function_base_impl import median
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding Data")

file_name='sales_data.csv'
if not os.path.exists(file_name):
    print(f"Error:File'{file_name}' not found")
    exit()



df = pd.read_csv(file_name)
print("Sucessfully Loaded")
print(f"Shape of the Dataset:Row:{df.shape[0]},columns:{df.shape[1]}")

print(df.head())
print(df.info())
print(df.describe())

print("Handling missing vaules:")

print(df.isnull().sum())

#With using median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)


median_spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)
print(median_spending)

#Distrbution Analysis

plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10,color='skyblue',edgecolor='black')
plt.title("Distrinution Amount")
plt.xlabel("Spending")
plt.ylabel("Number of customer")
plt.show()
#Correlational Matrix

correlation=df.corr(numeric_only=True)
print(correlation)

print("Plotting Correlation Matrix")
plt.figure(figsize=(7,4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("Find the Outliers is age")
outliers=df[df['Age']>100]
print("Found Outliers {s} :")
print(outliers)