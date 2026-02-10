import pandas as pd

df = pd.read_csv("datasets_macro_training5/global_dataset.csv")

df = df.rename(columns={
    'label': 'y_class',
    'tp_pct': 'y_tp',
    'sl_pct': 'y_sl'
})

df.to_csv("datasets_macro_training5/global_dataset_fixed.csv", index=False)
print("✅ Saved fixed dataset as global_dataset_fixed.csv")
