import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class TransformerPredictorModel(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, dropout=0.0, pred_len=15):
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
    def __init__(self , model=TransformerPredictorModel(pred_len=15) ,  params  = {'lr' : 1e-3 ,'loss': nn.MSELoss()}):
        super().__init__()
        self.model = model
        self.criterion = params['loss']
        self.params = params
        self.indx = 0
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('lr', lr, on_step=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("val_loss", loss)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, threshold =1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to watch
                "interval": "epoch",  # Check every epoch
                "frequency": 1,
            },
        }

