#import torch

#path = r"C:\users\natec\models_entry_1h\BTC_USD_entry.pt"

import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

# Allow loading of scaler and numpy scalar
torch.serialization.add_safe_globals([StandardScaler, np._core.multiarray.scalar])

# Path to your TPSL model
path = r"C:\users\natec\models_tp_sl_1h\BTC_USD_tp_sl.pt"

checkpoint = torch.load(path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

print("\n=== Model Parameter Shapes ===")
for k, v in state_dict.items():
    print(f"{k}: {tuple(v.shape)}")
