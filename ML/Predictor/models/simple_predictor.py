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

