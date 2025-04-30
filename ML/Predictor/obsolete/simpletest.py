import os
import inspect
import torch
import pytorch_lightning as pl
#import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import MAE , QuantileLoss
# 3. Download and Prepare Data
# For this example, we'll use Apple's stock data:
#

def get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 ,     batch_size = 64 ):
    data = pd.read_csv(os.path.join(datadir, 'train_stocks.csv'))
    #data["time_idx"] = data["date"].dt.year * * + data["date"].dt.monthdata["time_idx"] -= data["time_idx"].min()


    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="log_close",
        group_ids=["stock_name"],
        min_encoder_length=0,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["stock_name"],
        time_varying_known_reals=["time_idx", "month", "day", 'open', 'high', 'low', 'close','volume'],
        time_varying_unknown_reals=["log_close"],
        target_normalizer=GroupNormalizer(groups=["stock_name"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)


    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


    return train_dataloader, val_dataloader , training

def run_training( train_dataloader, val_dataloader , training):
    # Create a comprehensive wrapper class
    # Create a cleaner wrapper class
    class TFTLightningWrapper(pl.LightningModule):
        def __init__(self, training_dataset, **kwargs):
            super().__init__()
            self.save_hyperparameters(ignore=["training_dataset"])
            self.training_dataset = training_dataset

            # Store parameters for later use
            self.learning_rate = kwargs.get("learning_rate", 0.01)
            self.reduce_on_plateau_patience = kwargs.get("reduce_on_plateau_patience", 0)

            # Create TFT model
            self.tft = TemporalFusionTransformer.from_dataset(
                training_dataset,
                **kwargs
            )

        def forward(self, x):
            # Simple pass-through
            return self.tft(x)

        def shared_step(self, batch, batch_idx, stage):
            # Directly compute prediction and loss
            x, y = batch
            prediction = self(x)

            # Calculate loss using the model's loss function
            loss_value = self.tft.loss.loss(prediction, y)

            # Log the loss
            self.log(f"{stage}_loss", loss_value, on_epoch=True, prog_bar=True)

            return loss_value

        def training_step(self, batch, batch_idx):
            return self.shared_step(batch, batch_idx, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_step(batch, batch_idx, "val")

        def predict_step(self, batch, batch_idx):
            x, _ = batch
            return self(x)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

            if self.reduce_on_plateau_patience > 0:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=self.reduce_on_plateau_patience,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val_loss"
                }
            return optimizer

    # Create model
    model = TFTLightningWrapper(
        training_dataset=training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        limit_val_batches=30,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    )

    # Fit model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    # # tft = TemporalFusionTransformer.from_dataset(
    # #     training,
    # #     learning_rate=0.03,
    # #     hidden_size=16,
    # #     attention_head_size=4,
    # #     dropout=0.1,
    # #     hidden_continuous_size=8,
    # #     output_size=7,  # quantiles
    # #     loss=MAE(),
    # # )
    #
    # # Assume training is your TimeSeriesDataSet
    # model = TemporalFusionTransformer.from_dataset(
    #     training,
    #     learning_rate=0.03,
    #     hidden_size=16,  # biggest influence network size
    #     attention_head_size=1,
    #     dropout=0.1,
    #     hidden_continuous_size=8,
    #     output_size=7,  # QuantileLoss has 7 quantiles by default
    #     loss=QuantileLoss(),
    #     log_interval=10,  # log example every 10 batches
    #     reduce_on_plateau_patience=4,  # reduce learning automatically
    # )
    # trainer = pl.Trainer(
    #     max_epochs=20,
    #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #     devices=1,  # number of GPUs or CPUs
    #     gradient_clip_val=0.1,
    #     limit_train_batches=30,
    #     limit_val_batches=30,
    #     callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    # )
    #
    # trainer.fit(model, train_dataloader, val_dataloader)
    #
    # #######
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    #
    # predictions = best_tft.predict(val_dataloader)
    # actuals = torch.cat([y for x, y in iter(val_dataloader)])
    # mae = MAE()(predictions, actuals)
    # print(f"Mean Absolute Error: {mae.item()}")

if __name__ == "__main__":
    datadir = 'C:/Users/dadab/projects/algotrading/data/training/dbsmall'
    train_dataloader, val_dataloader, training = get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 ,     batch_size = 64 )
    run_training(train_dataloader, val_dataloader, training)