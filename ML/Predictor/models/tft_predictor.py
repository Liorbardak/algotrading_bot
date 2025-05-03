import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss , SMAPE
from pytorch_forecasting.data import GroupNormalizer


class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, training_dataset, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["training_dataset"])
        self.training_dataset = training_dataset

        # Store parameters for later use
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.reduce_on_plateau_patience = kwargs.get("reduce_on_plateau_patience", 0)

        # Create TFT model
        self.tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **kwargs
        )

    def forward(self, x):
        # Simple pass-through
        return self.tft(x)

    def shared_step(self, batch, batch_idx, stage):
        # Directly compute prediction and loss
        x, y = batch
        prediction = self(x)

        # Calculate loss using the model's loss function
        loss_value = self.tft.loss.loss(prediction[0], y[0])
        loss_value = loss_value.mean()
        # Log the loss
        self.log(f"{stage}_loss", loss_value.mean(), on_epoch=True, prog_bar=True)

        return loss_value

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.reduce_on_plateau_patience > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=self.reduce_on_plateau_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        return optimizer