import torch
import pandas as pd
from models.transformer_predictor import TransformerPredictorModel
from loaders.dataloaders import get_loaders
from lightingwraper import  LitStockPredictor

# Load model
quantiles = [0.1, 0.5, 0.9]
model = LitStockPredictor.load_from_checkpoint("stock_predictor.ckpt", input_dim=1, quantiles=quantiles)
model.eval()

# Sample input data: a price window (e.g., last 30 closing prices)
recent_prices = [160.3, 159.8, 160.9, ...]  # Replace with your real data
window = torch.tensor(recent_prices, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # shape (1, seq_len, 1)

# Predict
with torch.no_grad():
    prediction = model(window)

# Output predictions for each quantile
q_preds = {f"q_{int(q * 100)}": float(p) for q, p in zip(quantiles, prediction[0])}
print(q_preds)