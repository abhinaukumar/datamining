import pandas as pd

df = pd.read_csv('mini_eicu_features.csv')
df = df.head(20000)
df.to_csv('mini_eicu_features_weight_1.csv')