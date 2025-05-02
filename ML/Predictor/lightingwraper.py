import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from models.transformer_predictor import TransformerPredictorModel
class LitStockPredictor(pl.LightningModule):
    def __init__(self , model=TransformerPredictorModel() ,  params  = {'lr' : 1e-4 ,'loss': nn.MSELoss()}):
        super().__init__()
        self.model = model
        self.criterion = params['loss']
        self.params = params
        self.indx = 0
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr', lr, on_step=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.indx % 10 == 0:
            self.print("train_loss", loss)

            self.indx = self.indx+1

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("val_loss", loss)
        if self.indx % 10 == 0:
            print("val_loss", loss)
            self.indx = self.indx+1

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
