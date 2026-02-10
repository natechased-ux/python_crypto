
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --- Transformer Model ---
class TransformerEntryModel(nn.Module):
    def __init__(self, input_dim, seq_len=48, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerEntryModel, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.class_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.tp_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.sl_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _generate_positional_encoding(self, d_model, seq_len):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.input_proj(x)
        pe = self.positional_encoding.to(x.device)
        x = x + pe[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        label_logits = self.class_head(x)
        tp_out = self.tp_head(x)
        sl_out = self.sl_head(x)
        return label_logits, tp_out, sl_out

# --- Example Dataset Loading and Training Setup (Placeholder) ---
def load_data(csv_path, sequence_length):
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col not in ['label', 'tp_pct', 'sl_pct']]
    X, y_label, y_tp, y_sl = [], [], [], []

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, "scaler.pkl")

    for i in range(len(df) - sequence_length):
        X.append(df[feature_cols].iloc[i:i+sequence_length].values)
        y_label.append(df['label'].iloc[i+sequence_length])
        y_tp.append(df['tp_pct'].iloc[i+sequence_length])
        y_sl.append(df['sl_pct'].iloc[i+sequence_length])

    X = torch.tensor(X, dtype=torch.float32)
    y_label = torch.tensor(y_label, dtype=torch.long)
    y_tp = torch.tensor(y_tp, dtype=torch.float32).unsqueeze(-1)
    y_sl = torch.tensor(y_sl, dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(X, y_label, y_tp, y_sl)

def train_model():
    sequence_length = 48
    input_dim = 97  # Replace with your actual feature count
    batch_size = 64
    num_epochs = 10

    dataset = load_data("your_dataset.csv", sequence_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerEntryModel(input_dim=input_dim, seq_len=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_cls, y_tp, y_sl in train_loader:
            optimizer.zero_grad()
            out_cls, out_tp, out_sl = model(X_batch)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss_tp = criterion_reg(out_tp, y_tp)
            loss_sl = criterion_reg(out_sl, y_sl)
            loss = loss_cls + loss_tp + loss_sl
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "transformer_model.pth")
    print("✅ Transformer model saved.")

if __name__ == "__main__":
    train_model()
