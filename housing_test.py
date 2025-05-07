import pandas as pd

dataset = pd.read_csv('data/housing.csv')

df = dataset.describe()

print(df)
