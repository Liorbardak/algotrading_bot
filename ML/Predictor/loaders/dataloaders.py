import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


class StockDataset(Dataset):
    def __init__(self, df, seq_len=60, pred_len=20):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []

        for ticker in df['stock_name'].unique():
            stock_df = df[df['stock_name'] == ticker].copy()
            scaler = MinMaxScaler()
            values = scaler.fit_transform(stock_df[['open', 'open', 'low', 'close', 'volume']])

            for i in range(len(values) - seq_len - pred_len):
                x = values[i:i + seq_len]
                y = values[i + seq_len:i + seq_len + pred_len, 3]  # Close price
                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=64 ):
    tr_data = pd.read_csv(os.path.join(datadir, 'train_stocks.csv'))
    train_dataset = StockDataset(tr_data ,  seq_len=max_encoder_length, pred_len=max_prediction_length)
    val_data = pd.read_csv(os.path.join(datadir, 'val_stocks.csv'))
    val_dataset = StockDataset(val_data,  seq_len=max_encoder_length, pred_len=max_prediction_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
