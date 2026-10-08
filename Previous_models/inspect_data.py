import pandas as pd

df=pd.read_csv("data/ai4i2020.csv")

print("Shape:")
print(df.shape)

print("Columns:")
print(df.columns)

print("first 5 rows:")
print(df.head())

print("Data types:")
print(df.dtypes)

print("Missing values:")
print(df.isnull().sum())

print("Duplicate values:")
print(df.duplicated().sum())

print("Target distribution:")
print(df["Machine failure"].value_counts())