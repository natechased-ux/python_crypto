import os
import glob
from train_tp_sl import train_tp_sl_model  # this is the updated function I gave you

# Paths
DATASET_PATH = "datasets_macro_training"  # folder with your coin CSVs
OUTPUT_PATH = "models_tp_sl_1h"           # folder to save TP/SL models

# Training settings
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 5  # early stopping patience

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    datasets = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
    print(f"📊 Found {len(datasets)} datasets to train TP/SL models.")

    for dataset in datasets:
        coin_name = os.path.basename(dataset).replace(".csv", "")
        print(f"\n🚀 Training TP/SL model for {coin_name}...")

        output_file = os.path.join(OUTPUT_PATH, f"{coin_name}_tp_sl.pt")

        try:
            train_tp_sl_model(
                dataset_path=dataset,
                output_path=output_file,
                seq_len=SEQ_LEN,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                patience=PATIENCE
            )
        except Exception as e:
            print(f"❌ Error training {coin_name}: {e}")

    print("\n🎯 All TP/SL models trained and saved.")

if __name__ == "__main__":
    main()
