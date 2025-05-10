import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch, math
from torch import nn, Tensor


class LinearPredictorModel(nn.Module):
    def __init__(self, input_dim=5, pred_len=15 , seq_len=60):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len),
        )

    def forward(self, x):

        out = self.regressor(x)
        return  out[:,-1,:]


#
# class SimplePredictor(pl.LightningModule):
#     def __init__(self , model=LinearPredictorModel(pred_len=15 ,seq_len=60 ) ,  params  = {'lr' : 1e-3 ,'loss': nn.MSELoss()}):
#         super().__init__()
#         self.model = model
#         self.criterion = params['loss']
#         self.params = params
#         self.indx = 0
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         preds = self(x)
#         loss = self.criterion(preds, y)
#         lr = self.trainer.optimizers[0].param_groups[0]['lr']
#
#         self.log('lr', lr, on_step=True, prog_bar=True, logger=True)
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         preds = self(x)
#         loss = self.criterion(preds, y)
#
#         self.log("val_loss", loss)
#
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])
#         #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold =1e-4)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss",  # Metric to watch
#                 "interval": "epoch",  # Check every epoch
#                 "frequency": 1,
#             },
#         }
#
#     def get_dataset_params(self):
#         return []
