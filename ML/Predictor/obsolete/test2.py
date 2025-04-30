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

def get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60):
    tr_data = pd.read_csv(os.path.join(datadir, 'train_stocks.csv'))
    train_dataset = StockDataset(tr_data ,  seq_len=max_encoder_length, pred_len=max_prediction_length)
    val_data = pd.read_csv(os.path.join(datadir, 'val_stocks.csv'))
    val_dataset = StockDataset(val_data,  seq_len=max_encoder_length, pred_len=max_prediction_length)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    return train_loader, val_loader

class TransformerModel(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, dropout=0.1, seq_len=60, pred_len=20):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # use last token's representation
        return self.regressor(x)


class LitStockPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerModel()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss)
        print("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss)
        print("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def run_training(datadir: str, outdir: str):
    checkpoints_path = os.path.join(outdir, 'checkpoints')
    log_path = os.path.join(outdir, 'logs')
    os.makedirs(checkpoints_path ,exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor="val_loss",       # metric to track
        mode="min",               # "min" if lower is better (e.g., loss)
        save_top_k=3,             # save only the best model
        filename="best-checkpoint",  # name of the file
        verbose=True
    )


    train_loader, val_loader= get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60)
    model = LitStockPredictor()
    #trainer = pl.Trainer(max_epochs=10, accelerator="auto")
    trainer = pl.Trainer(max_epochs=2, accelerator="gpu", devices=1 ,     callbacks=[checkpoint_callback],
                         default_root_dir=log_path,
                         )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    datadir = 'C:/Users/dadab/projects/algotrading/data/training/dbsmall'
    outdir = "C:/Users/dadab\projects/algotrading/training/test"


    run_training(datadir , outdir)