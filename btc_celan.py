import os
import pandas as pd

# Path to folder with all your coin CSVs
DATA_DIR = "datasets_macro_training5"

# Target columns we want to remove from *every* coin file
cols_to_remove = ["btc_label", "btc_tp_pct", "btc_sl_pct"]

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)

        # Find any of the unwanted columns present in this file
        drop_cols = [c for c in cols_to_remove if c in df.columns]
        if drop_cols:
            print(f"Cleaning {filename} — removing {drop_cols}")
            df.drop(columns=drop_cols, inplace=True)
            df.to_csv(file_path, index=False)
        else:
            print(f"{filename} — no columns to drop")

print("✅ All CSVs cleaned of BTC target columns.")
