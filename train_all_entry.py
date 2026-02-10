import os
import subprocess

# === CONFIG ===
DATASET_DIR = "datasets_macro_training"   # Folder with your macro-enhanced CSVs
MODEL_DIR = "models_entry_macro"          # Where to save models
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3

# Make sure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Get all CSV datasets in the folder
datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]

print(f"Found {len(datasets)} datasets to train.")

# Loop through and train each one
for csv_file in datasets:
    coin_name = csv_file.replace(".csv", "").upper()
    dataset_path = os.path.join(DATASET_DIR, csv_file)
    model_path = os.path.join(MODEL_DIR, f"{coin_name}_entry.pt")

    print(f"\n🚀 Training entry model for {coin_name}...")
    cmd = [
        "py", "train_entry_lstm.py",
        "--dataset", dataset_path,
        "--output", model_path,
        "--seq_len", str(SEQ_LEN),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR)
    ]
    subprocess.run(cmd)

print("\n✅ All entry models trained and saved.")
