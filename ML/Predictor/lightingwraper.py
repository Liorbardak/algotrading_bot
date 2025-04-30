import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


class LitStockPredictor(pl.LightningModule):
    def __init__(self , model, loss, params  = {'lr' : 1e-3}):
        super().__init__()
        self.model = model
        self.criterion = loss
        self.params = params
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        #self.log("train_loss", loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.print("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss)
        print("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to watch
                "interval": "epoch",  # Check every epoch
                "frequency": 1,
            },
        }
