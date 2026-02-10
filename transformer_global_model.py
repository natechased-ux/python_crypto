import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerWithCoin(nn.Module):
    def __init__(self, input_dim, coin_count, d_model=96, nhead=4, num_layers=1, coin_emb_dim=8):
        super().__init__()
        self.coin_emb = nn.Embedding(coin_count, coin_emb_dim)
        self.input_proj = nn.Linear(input_dim + coin_emb_dim, d_model)
        self.posenc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 3))
        self.tp_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sl_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, coin_id):
        # x shape: (batch, seq, features)
        coin_feat = self.coin_emb(coin_id).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, coin_feat], dim=-1)
        x = self.input_proj(x)
        x = self.posenc(x)
        x = self.encoder(x)
        h = x[:, -1, :]
        return self.cls_head(h), self.tp_head(h), self.sl_head(h)
