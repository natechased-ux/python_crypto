import pandas as pd

df = pd.read_csv("datasets_macro_training/ETH_USD.csv")
cols_needed = ["time", "close", "high", "low"]
df_small = df[cols_needed]
df_small.to_csv("ETH_USD_small.csv", index=False)
print(f"Saved reduced dataset with {len(df_small)} rows to ETH_USD_small.csv")
