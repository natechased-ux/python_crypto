import os
import glob
import subprocess

# Config
DATASET_FOLDER = "datasets_macro_training"
TRAIN_SCRIPT = "train_entry_lstm.py"
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001

def main():
    datasets = sorted(glob.glob(os.path.join(DATASET_FOLDER, "*.csv")))
    print(f"📊 Found {len(datasets)} datasets to train.\n")

    if not datasets:
        print("❌ No datasets found!")
        return

    for dataset_path in datasets:
        coin_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"\n🚀 Training {coin_name}...")

        model_save_path = os.path.join("models_entry_1h", f"{coin_name}_entry.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        subprocess.run([
            "python", TRAIN_SCRIPT,
            "--dataset", dataset_path,
            "--seq_len", str(SEQ_LEN),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--model_save_path", model_save_path
        ])

    print("\n🎯 All training complete!")

if __name__ == "__main__":
    main()
