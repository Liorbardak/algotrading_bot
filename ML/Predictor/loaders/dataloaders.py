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
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

class StockDataset(Dataset):
    def __init__(self, df, seq_len=60, pred_len=2 , step = 1 , get_meta = False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []
        self.meta = []

        for ticker in df['stock_name'].unique():
            stock_df = df[df['stock_name'] == ticker].copy()

            values = stock_df[['open', 'high', 'low', 'close', 'volume']].values
            dates = stock_df['date'].values
            for i in range(1,len(values) - seq_len - pred_len, step):
                x = values[i:i + seq_len]
                y = values[i + seq_len:i + seq_len + pred_len, 3]  # prediction GT
                #y = values[i + seq_len-1:i + seq_len + pred_len-1, 3]  # prediction GT , with last sample
                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32)))
                if get_meta:
                     self.meta.append({'stock' : ticker,
                                       'date' : dates[ i + seq_len-1],
                                       'time_index': i + seq_len-1})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_meta(self, idx):
        return self.meta[idx]


def get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=32 , step = 1 , loader_type : str = 'tft' ):

    train_loader = get_loader(datadir, 'train_stocks.csv',max_prediction_length,max_encoder_length,batch_size, step ,shuffle=True , loader_type=loader_type )
    val_loader = get_loader(datadir, 'val_stocks.csv', max_prediction_length, max_encoder_length, batch_size,step,
                              shuffle=False , loader_type=loader_type)

    return train_loader, val_loader

def get_loader(datadir,filename, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=32,step=1, shuffle=True,loader_type : str = 'tft' , get_meta = True):
    data = pd.read_csv(os.path.join(datadir, filename))
    if loader_type  != 'tft':
        dataset = StockDataset(data ,  seq_len=max_encoder_length, pred_len=max_prediction_length,step=step , get_meta=get_meta)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # does not work with TimeSeriesDataSet
    else:
        dataset = TimeSeriesDataSet(
                data=data,
                time_idx="time_idx",
                target="close",
                group_ids=["stock_id"],
                min_encoder_length=max_encoder_length,  # Minimum history length
                max_encoder_length=max_encoder_length,  # Maximum history length
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                static_categoricals=["stock_name"],
                time_varying_known_categoricals=[],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_categoricals=[],
                time_varying_unknown_reals=[
                    "open", "high", "low", "close", "volume",
                ],
                target_normalizer=GroupNormalizer(
                    groups=["stock_id"], transformation="softplus"
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,  # Allow gaps in time series data
        )
        loader = dataset.to_dataloader(batch_size=batch_size, shuffle=shuffle)

    return loader
