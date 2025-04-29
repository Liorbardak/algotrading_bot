import torch
import pytorch_lightning as pl
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import MAE
# 3. Download and Prepare Data
# For this example, we'll use Apple's stock data:
#
# python
# Copy
# Edit
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01', progress=False)
data['Date'] = data.index
data['time_idx'] = np.arange(len(data))
data['log_close'] = np.log(data['Close'])

# Add additional features
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['year'] = data['Date'].dt.year
#4. Define Dataset for PyTorch Forecasting python Copy Edit
max_prediction_length = 30  # forecast 30 days
max_encoder_length = 60  # use 60 days of history
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="log_close",
    group_ids=["year"],
    min_encoder_length=0,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["year"],
    time_varying_known_reals=["time_idx", "month", "day"],
    time_varying_unknown_reals=["log_close"],
    target_normalizer=GroupNormalizer(groups=["year"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


trainer = pl.Trainer(
    max_epochs=20,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    limit_val_batches=30,
    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # quantiles
    loss=MAE(),
)

trainer.fit(tft, train_dataloader, val_dataloader)
# 6. Evaluate the Model
# python
# Copy
# Edit
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions = best_tft.predict(val_dataloader)
actuals = torch.cat([y for x, y in iter(val_dataloader)])
mae = MAE()(predictions, actuals)
print(f"Mean Absolute Error: {mae.item()}")