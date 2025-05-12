import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
from loaders.dataloaders import get_loader

# ----------------------------
# Generate synthetic signal
# For simplicity, create 5 different sinusoidal signals
# ----------------------------
signal = np.vstack([np.sin(np.linspace(0, 100, 500)),
                    np.cos(np.linspace(0, 100, 500)),
                    np.tan(np.linspace(0, 100, 500))[:500],  # Capped to avoid infinity
                    np.sin(np.linspace(0, 50, 500)),
                    np.cos(np.linspace(0, 50, 500))]).T  # 500 samples, 5 features

# ----------------------------
# Dataset
# ----------------------------
class SequenceToSequenceDataset(Dataset):
    def __init__(self, signal, input_len=60, pred_len=15):
        self.signal = torch.tensor(signal, dtype=torch.float32)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.signal) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.signal[idx:idx + self.input_len]  # [60, 5]
        y = self.signal[idx + self.input_len:idx + self.input_len + self.pred_len, 0]  # [15] (predict only the first feature)
        return x, y  # x: [60, 5], y: [15]

# ----------------------------
# LSTM Model
# ----------------------------
class Seq2SeqLSTM(L.LightningModule):
    def __init__(self, input_size=5, hidden_size=32, output_len=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)  # Output 15 values (predictions for the first feature)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: [1, batch, hidden]
        out = self.fc(hn[-1])      # [batch, 15]
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        print(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# ----------------------------
# Training
# ----------------------------
if __name__ == "__main__":
    dataset = SequenceToSequenceDataset(signal, input_len=60, pred_len=15)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    loader  = get_loader('C:/Users/dadab/projects/algotrading/data/training/snp_overfit/', 'train_stocks.csv', max_prediction_length = 15 , max_encoder_length = 60
                                   , batch_size=16 , shuffle=False, get_meta = True , loader_type = 'regressor' )


    model = Seq2SeqLSTM(input_size=5, output_len=15)

    trainer = L.Trainer(max_epochs=100, logger=False, enable_checkpointing=False)
    trainer.fit(model, loader)