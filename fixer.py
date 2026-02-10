import torch
import os
from torch.serialization import add_safe_globals
from sklearn.preprocessing import StandardScaler

# ✅ Allow safe unpickling of sklearn scaler objects
add_safe_globals([StandardScaler])

model_dir = "models_lstm2"
output_dir = "models_lstm_cleaned"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(model_dir):
    if filename.endswith(".pth"):
        full_path = os.path.join(model_dir, filename)
        print(f"Processing: {filename}")

        try:
            checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue

        # Extract weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            weights = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            print(f"⚠️ Skipping {filename} — doesn't contain model_state_dict")
            continue
        else:
            weights = checkpoint  # Already clean

        # Save just the model weights
        cleaned_path = os.path.join(output_dir, filename)
        torch.save(weights, cleaned_path)
        print(f"✅ Cleaned and saved: {cleaned_path}")
