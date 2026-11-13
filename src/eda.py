import pandas as pd

df = pd.read_csv(r"E:\GitHub\Titanic-ids\data\train.csv")

print("Shape:", df.shape)

print("\nData types and non-null counts:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Survived"].value_counts(normalize=True))

print("\nSurvival rate by sex:")
print(df.groupby("Sex")["Survived"].mean())

print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass")["Survived"].mean())