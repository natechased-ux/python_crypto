import torch
from torch.utils.data import Dataset

class StreamingWindowDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len):
        self.df = df.reset_index(drop=True)
        self.features = feature_cols
        self.seq_len = seq_len
        self.valid_indices = []

        for i in range(len(df) - seq_len - 1):
            if len(set(df['coin_id_code'].iloc[i:i+seq_len])) == 1:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        window = self.df.iloc[i:i+self.seq_len]
        target = self.df.iloc[i+self.seq_len]

        X = torch.tensor(window[self.features].values, dtype=torch.float32)
        coin_id = torch.tensor(window['coin_id_code'].iloc[0], dtype=torch.long)
        y_class = torch.tensor(target['y_class'], dtype=torch.long)
        y_tp = torch.tensor(target['y_tp'], dtype=torch.float32)
        y_sl = torch.tensor(target['y_sl'], dtype=torch.float32)
        return X, coin_id, y_class, y_tp.unsqueeze(0), y_sl.unsqueeze(0)
