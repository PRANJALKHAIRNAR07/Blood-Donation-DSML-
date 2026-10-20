
import pandas as pd

# Load the dataset correctly
df = pd.read_csv("data/transfusion.data", sep=",", skipinitialspace=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Rename the target column for convenience
df.columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "Time",
    "Donated_Blood"
]

print("\nDonation outcome distribution:")
print(df["Donated_Blood"].value_counts())

print("\nDataset information:")
print(df.info())
