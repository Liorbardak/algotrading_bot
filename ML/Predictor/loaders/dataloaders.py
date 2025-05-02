import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib

class StockDataset(Dataset):
    def __init__(self, df, seq_len=60, pred_len=2):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []
        self.meta = []

        for ticker in df['stock_name'].unique():
            stock_df = df[df['stock_name'] == ticker].copy()

            values = stock_df[['open', 'high', 'low', 'close', 'volume']].values

            for i in range(len(values) - seq_len - pred_len):
                x = values[i:i + seq_len]
                y = values[i + seq_len:i + seq_len + pred_len, 3]  # Close price
                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32)))
                self.meta.append({'stock' : ticker,
                                  'time_index': i + seq_len})
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_meta(self, idx):
        return self.meta[idx]


def get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=64 , output_file = None ):

    train_loader = get_loader(datadir, 'train_stocks.csv',max_prediction_length,max_encoder_length,batch_size, shuffle=True, output_file=output_file )
    val_loader = get_loader(datadir, 'val_stocks.csv', max_prediction_length, max_encoder_length, batch_size,
                              shuffle=False)

    return train_loader, val_loader

def get_loader(datadir,filename, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=64, shuffle=True , output_file = None ):
    tr_data = pd.read_csv(os.path.join(datadir, filename))
    dataset = StockDataset(tr_data ,  seq_len=max_encoder_length, pred_len=max_prediction_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=15 , persistent_workers=True)
    return loader
