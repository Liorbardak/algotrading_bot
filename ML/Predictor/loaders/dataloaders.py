import os
import torch
import copy
import torch.nn as nn
import pytorch_lightning as pl
from fontTools.misc.bezierTools import epsilon
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from scipy.ndimage import median_filter

class StockDataset(Dataset):
    def __init__(self, df, seq_len=60, pred_len=2 , step = 1 , get_meta = False,per_filter = 'medfilt', features=['open', 'high', 'low', 'close', 'volume'] ,
                 features_to_normalize = ['open', 'high', 'low', 'close', 'volume','ma20','ma50']):

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.per_filter = per_filter
        self.samples = []
        self.meta = []
        reference_feature=   'close'
        features_to_normalize_idx = [features.index(f) for f in features_to_normalize if f in features]
        reference_feature_idx = features.index(reference_feature)

        for ticker in df['ticker'].unique():
            stock_df = df[df['ticker'] == ticker].copy()

            values = stock_df[features].values
            dates = stock_df['date'].values
            for i in range(1,len(values) - seq_len - pred_len, step):
                if  self.per_filter == 'medfilt':
                    vals = copy.copy(values[i:i + seq_len + pred_len])
                    x =  median_filter(vals[:seq_len],  size=(3,1), mode='nearest')
                    y = median_filter(vals[seq_len:seq_len + pred_len, reference_feature_idx],  size=3, mode='nearest')
                else:
                    x = copy.copy(values[i:i + seq_len])
                    y =  copy.copy(values[i + seq_len:i + seq_len + pred_len, reference_feature_idx])  # prediction GT

                # Normalize by the  reference_feature (close price) of the last time in the input
                reference = x[-1, reference_feature_idx]
                norm_factor =  (reference+  1e-5)
                # normalize all related features
                x[:, features_to_normalize_idx] = x[:, features_to_normalize_idx] / norm_factor

                y = y / norm_factor
                # let the prediction target to be the offset respect the last normalized price (== 1)   TODO - revisit
                y = y-1


                self.samples.append((torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32)))
                if get_meta:
                     self.meta.append({'stock' : ticker,
                                       'date' : dates[ i + seq_len-1],
                                       'time_index': i + seq_len-1,
                                       'norm_factor' : norm_factor
                                       }
                                      )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_meta(self, idx):
        return self.meta[idx]


def get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=32 , step = 1 , loader_type : str = 'tft',
                features=["open", "high", "low", "close", "volume"]):
    if loader_type != 'tft':
        train_loader = get_loader(datadir, 'train_stocks.csv',max_prediction_length,max_encoder_length,batch_size, step ,shuffle=True , loader_type=loader_type,features=features )
        val_loader = get_loader(datadir, 'val_stocks.csv', max_prediction_length, max_encoder_length, batch_size,step,
                              shuffle=False , loader_type=loader_type, features=features)
    else:
        # Need to split validation/train by time
        df_train = pd.read_csv(os.path.join(datadir, 'train_stocks.csv'))
        df_val = pd.read_csv(os.path.join(datadir, 'val_stocks.csv'))
        data = pd.concat([df_train, df_val]).reset_index(drop=True)
        training_cutoff = data["time_idx"].max() - max_prediction_length
        training = TimeSeriesDataSet(
            data= data[data.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="close",
            group_ids=["stock_id"],
            min_encoder_length=max_encoder_length,  # Minimum history length
            max_encoder_length=max_encoder_length,  # Maximum history length
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["ticker"],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=features,
            target_normalizer=GroupNormalizer(
                groups=["stock_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,  # Allow gaps in time series data

            categorical_encoders={
                "group_id": NaNLabelEncoder(add_nan=True),  # allow unknown categories - for inference over new categories
                "ticker_id": NaNLabelEncoder(add_nan=True),
                "ticker": NaNLabelEncoder(add_nan=True),
                "__group_id__ticker_id": NaNLabelEncoder(add_nan=True),  # optional: auto-created if group_ids used
            }
        )
        train_loader = training.to_dataloader(batch_size=batch_size, shuffle=True)
        validation = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True)
        val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)



    return train_loader, val_loader

def get_loader(datadir,filename, max_prediction_length = 20 , max_encoder_length = 60 , batch_size=32,step=1,
               shuffle=True,loader_type : str = 'tft' , get_meta = True, features=["open", "high", "low", "close", "volume"]):
    data = pd.read_csv(os.path.join(datadir, filename))
    if loader_type  != 'tft':
        dataset = StockDataset(data ,  seq_len=max_encoder_length, pred_len=max_prediction_length,step=step , get_meta=get_meta,
                               features=features)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # Get the training loader first

        dataset = TimeSeriesDataSet(
            data=data,
            time_idx="time_idx",
            target="close",
            group_ids=["stock_id"],
            min_encoder_length=max_encoder_length,  # Minimum history length
            max_encoder_length=max_encoder_length,  # Maximum history length
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["ticker"],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=features,
            target_normalizer=GroupNormalizer(
                groups=["stock_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,  # Allow gaps in time series data

            categorical_encoders={
                "group_id": NaNLabelEncoder(add_nan=True),
                # allow unknown categories - for inference over new categories
                "ticker_id": NaNLabelEncoder(add_nan=True),
                "ticker": NaNLabelEncoder(add_nan=True),
                "__group_id__ticker_id": NaNLabelEncoder(add_nan=True),  # optional: auto-created if group_ids used
            }
        )
        # Get the inference loader
        loader = dataset.to_dataloader(batch_size=batch_size, shuffle=shuffle)
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



    return loader
