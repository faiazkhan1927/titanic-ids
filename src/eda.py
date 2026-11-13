
import pandas as pd

# Load training dataset
df = pd.read_csv("../data/train.csv")

# Basic dataset information
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Records ---")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe(include="all"))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Distribution ---")
print(df["Survived"].value_counts())

print("\n--- Target Distribution (%) ---")
print(df["Survived"].value_counts(normalize=True) * 100)
