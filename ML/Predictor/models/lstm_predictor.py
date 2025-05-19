import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
# ----------------------------
# LSTM Model
# ----------------------------
class Seq2SeqLSTM(L.LightningModule):
    def __init__(self, input_size=5, hidden_size=32, output_len=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)  # Output  values (predictions for the first feature)


    def forward(self, x):
        # x: [ batch, seq_len,  features]
        _, (hn, _) = self.lstm(x)  # hn: [1, batch, hidden]
        out = self.fc(hn[-1])      # [batch, output_len]
        return out

class Seq2SeqLSTM2(L.LightningModule):
    def __init__(self, input_size=5, hidden_size=64, num_layers = 2, output_len=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),  # First linear layer
            nn.ReLU(),  # Non-linear activation
            nn.Linear(hidden_size//2, output_len)  # Second linear layer
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: [1, batch, hidden]
        out = self.fc(hn[-1])      # [batch, outputlen]
        return out

