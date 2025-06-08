import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch, math
from torch import nn, Tensor

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. Mask out future positions."""
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=5, d_model=64, nhead=4, num_layers=2, dropout=0.1 ,  pred_len=15):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:seq_len]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, embed_dim]

        # No need for this mask , since we are doing batch forecasting with complete historical sequences
        # mask = generate_square_subsequent_mask(seq_len).to(x.device)
        # x = self.transformer_encoder(x, mask)

        x = self.transformer_encoder(x)

        x = x[-1]  # last time step
        return self.decoder(x)




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)



class TransformerPredictorModel(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=1,dim_feedforward=2048, dropout=0.0, pred_len=15 , seq_len=60):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, pred_len)

    def generate_causal_mask(self, seq_len):
        # [seq_len, seq_len] with -inf above the diagonal
        return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch, seq, d_model]
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)

        out = self.transformer_encoder(x, mask=causal_mask)
        out = self.output_layer(out)[:,-1,:]
        return  out




class LitStockPredictor(pl.LightningModule):
   # def __init__(self , model=TransformerPredictorModel(pred_len=15 ,seq_len=60 ) ,  params  = {'lr' : 1e-3 ,'loss':  nn.MSELoss()}):
    def __init__(self, model=TimeSeriesTransformer(pred_len=15),params={'lr': 1e-3, 'loss': nn.MSELoss()}):
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

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("val_loss", loss)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold =1e-4 , min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to watch
                "interval": "epoch",  # Check every epoch
                "frequency": 1,
            },
        }

    def get_dataset_params(self):
        return []
