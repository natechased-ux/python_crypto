import os
import glob
import subprocess

DATASET_DIR = "datasets_macro_training"  # your hourly datasets
MODEL_DIR = "models_tp_sl_1h"
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)

datasets = glob.glob(f"{DATASET_DIR}/*.csv")
print(f"📊 Found {len(datasets)} datasets to train.")

for dataset in datasets:
    coin_name = os.path.splitext(os.path.basename(dataset))[0]
    print(f"\n🚀 Training TP/SL model for {coin_name}...")
    model_path = f"{MODEL_DIR}/{coin_name}_tp_sl.pt"

    subprocess.run([
        "python", "train_tp_sl_lstm.py",
        "--dataset", dataset,
        "--output", model_path,
        "--seq_len", str(SEQ_LEN),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR)
    ])
