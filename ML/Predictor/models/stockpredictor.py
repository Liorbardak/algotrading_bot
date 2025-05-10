import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
# Time Series Transformer Model
class StockTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_size=32, num_layers=2, num_heads=2, dropout=0.0, pred_len=5):
        super(StockTransformer, self).__init__()

        self.encoder_embedding = nn.Linear(input_dim, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Prediction head
        self.fc_out = nn.Linear(hidden_size, 1)  # Predict only the target feature
        self.pred_len = pred_len

    def get_dataset_params(self):
        return []

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # Project input to hidden dimension
        x = self.encoder_embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer
        transformer_output = self.transformer_encoder(x)

        # Extract the last time step's output
        last_step = transformer_output[:, -1:, :]

        # Expand last step for prediction length
        repeated_last = last_step.repeat(1, self.pred_len, 1)

        # Project to output dimension
        output = self.fc_out(repeated_last)
        return output.squeeze(-1)  # Shape: [batch_size, pred_len, 1]
