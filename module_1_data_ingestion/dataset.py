import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv")
print(df.shape)      # (10459, 24)
print(df.columns)    # All credit bureau numerical features
print(df.head())