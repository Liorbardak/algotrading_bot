import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss , SMAPE
from pytorch_forecasting.data import GroupNormalizer


class StockPricePredictor:
    def __init__(self,
                 # ticker_symbols: List[str],
                 # start_date: str,
                 # end_date: str,
                 forecast_horizon: int = 5,
                 context_length: int = 30,
                 train_val_split: float = 0.8):
        """
        Initialize the Stock Price Predictor

        Args:
            ticker_symbols: List of stock ticker symbols to fetch data for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            forecast_horizon: Number of days to forecast
            context_length: Number of days to use as context
            train_val_split: Ratio of training data vs validation data
        """
        # self.ticker_symbols = ticker_symbols
        # self.start_date = start_date
        # self.end_date = end_date
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.train_val_split = train_val_split

        # Placeholder for data and model
        self.data = None
        self.training = None
        self.validation = None
        self.model = None
        self.trainer = None
        self.prediction_data = None

    def fetch_data(self):
        """
        Fetch stock price data using yfinance API
        """
        # Download stock data
        # all_data = []
        # inputdir = 'C:/Users/dadab/projects/algotrading/data/tickers'
        # for ticker in self.ticker_symbols:
        #     try:
        #
        #         #stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
        #         stock_data =  pd.read_excel(os.path.join(inputdir, ticker, 'stockPrice.xlsx'), engine='openpyxl')
        #         # Extract relevant columns and prepare data
        #         stock_data.rename(columns={'1. open':'open', '2. high': 'high', '3. low' : 'low' ,'close': 'close' , '5. volume' : 'volume', 'Date' : 'date'}, inplace=True)
        #         df = stock_data[['open', 'high', 'low', 'close', 'volume','date']].copy()
        #
        #         # # Calculate additional technical indicators
        #         # # Moving averages
        #         # df['MA5'] = df['Close'].rolling(window=5).mean()
        #         # df['MA20'] = df['Close'].rolling(window=20).mean()
        #         #
        #         # # RSI (Relative Strength Index)
        #         # delta = df['Close'].diff()
        #         # gain = delta.where(delta > 0, 0).fillna(0)
        #         # loss = -delta.where(delta < 0, 0).fillna(0)
        #         # avg_gain = gain.rolling(window=14).mean()
        #         # avg_loss = loss.rolling(window=14).mean()
        #         # rs = avg_gain / avg_loss
        #         # df['RSI'] = 100 - (100 / (1 + rs))
        #         #
        #         # # MACD (Moving Average Convergence Divergence)
        #         # df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        #         # df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        #         # df['MACD'] = df['EMA12'] - df['EMA26']
        #         # df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        #         #
        #         # # Bollinger Bands
        #         # df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        #         # df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        #         # df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        #         #
        #         # # Percentage price change
        #         # df['Price_Change'] = df['Close'].pct_change() * 100
        #
        #         # Reset index to make the date a column
        #         df.reset_index(inplace=True)
        #
        #         # Add ticker column
        #         df['stock_name'] = ticker
        #
        #         # Add to list
        #         all_data.append(df)
        #
        #         print(f"Successfully downloaded data for {ticker}")
        #     except Exception as e:
        #         print(f"Error downloading data for {ticker}: {e}")
        #
        # # Combine all data
        # self.data = pd.concat(all_data, ignore_index=True)
        # #self.data.rename(columns={'Date': 'date'}, inplace=True)
        #
        # # Fill NA values
        # self.data = self.data.fillna(0)
        #
        # # Create time index - ensure each ticker starts from 0
        # self.data = self.data.sort_values(['stock_name', 'date'])
        # self.data['time_idx'] = self.data.groupby('stock_name').cumcount()
        #
        #
        # # Drop rows with missing values
        # self.data = self.data.dropna()
        #
        # print(f"Data shape: {self.data.shape}")

        #self.data = pd.read_csv("C:/Users/dadab/projects/algotrading/data/training/samllbb2/train_stocks.csv")
        self.data= pd.read_csv("C:/Users/dadab/projects/algotrading/data/training/good1/train_stocks.csv")

    def prepare_datasets(self):
        """
        Prepare TimeSeriesDataSet for training and validation
        """

        # Determine cutoff point between train and validation
        train_cutoff = int(5000 * self.train_val_split)

        # Create training dataset
        self.training = TimeSeriesDataSet(
            data=self.data[:train_cutoff],
            time_idx="time_idx",
            target="close",
            group_ids=["stock_name"],
            min_encoder_length=self.context_length // 2,  # Minimum history length
            max_encoder_length=self.context_length,  # Maximum history length
            min_prediction_length=1,
            max_prediction_length=self.forecast_horizon,
            static_categoricals=["stock_name"],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "open", "high", "low", "close", "volume",
            ],
            target_normalizer=GroupNormalizer(
                groups=["stock_name"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True, # Allow gaps in time series data
        )

        # Create validation dataset using parameters from training dataset
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, self.data[train_cutoff:], predict=True, stop_randomization=True
        )

        # Create dataloaders
        self.train_dataloader = self.training.to_dataloader(
            batch_size=32, num_workers=0, shuffle=True
        )

        self.val_dataloader = self.validation.to_dataloader(
            batch_size=32, num_workers=0, shuffle=False
        )

        # Create dataset for prediction
        self.prediction_data = TimeSeriesDataSet.from_dataset(
            self.training, self.data, predict=True, stop_randomization=True
        )

        self.prediction_dataloader = self.prediction_data.to_dataloader(
            batch_size=32, num_workers=0, shuffle=False
        )

    def create_model(self):
        """
        Create Temporal Fusion Transformer model
        """

        class TFTLightningWrapper(pl.LightningModule):
            def __init__(self, training_dataset = None, **kwargs):
                super().__init__()
                self.save_hyperparameters(ignore=["training_dataset"])
                self.training_dataset = training_dataset

                # Store parameters for later use
                self.learning_rate = kwargs.get("learning_rate", 0.01)
                self.reduce_on_plateau_patience = kwargs.get("reduce_on_plateau_patience", 0)

                # Create TFT model
                if training_dataset is not None:
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
                loss_value = self.tft.loss.loss(prediction[0], y[0])
                loss_value = loss_value.mean()
                # Log the loss
                self.log(f"{stage}_loss", loss_value.mean(), on_epoch=True, prog_bar=True)

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



        # Define the loss function
        #loss = QuantileLoss(reduction="mean")
        loss =SMAPE(reduction="mean")
        self.model = TFTLightningWrapper(
        training_dataset=self.training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=1,
        loss=loss,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
        # # Set up the TFT model
        # self.model = TemporalFusionTransformer.from_dataset(
        #     self.training,
        #     learning_rate=0.001,
        #     hidden_size=64,
        #     attention_head_size=4,
        #     dropout=0.1,
        #     hidden_continuous_size=32,
        #     loss=loss,
        #     log_interval=10,
        #     reduce_on_plateau_patience=4,
        # )

        # Set up early stopping and model checkpointing
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, verbose=True, mode="min"
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )

        # Set up logger
        logger = TensorBoardLogger("lightning_logs", name="stock-forecast")

        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=50,
            # accelerator="gpu" if torch.cuda.is_available() else "cpu",
            # devices=1 if torch.cuda.is_available() else 0,
            accelerator="gpu",
            devices= 1,
            gradient_clip_val=0.1,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
        )

    def train_model(self):
        """
        Train the model
        """
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )

    def make_predictions(self, return_x: bool = False):
        """
        Make predictions using the trained model

        Args:
            return_x: If True, return input data along with predictions

        Returns:
            Predictions dataframe
        """
        # Make predictions
        predictions = self.model.predict(
            self.prediction_dataloader,
            return_x=return_x,
            mode="raw"
        )

        if return_x:
            # Extract inputs and predictions
            x, output = predictions
            predictions_df = pd.DataFrame(
                output[0].cpu().numpy(),
                columns=[f"Quantile_{q}" for q in self.model.loss.quantiles]
            )

            # Get the actual values
            actuals = x["encoder_target"].cpu().numpy()

            # Get the dates and tickers
            dates = x["decoder_time_idx"].cpu().numpy()
            tickers = np.array([self.training.index_to_group_mapping[int(i)] for i in x["groups"].cpu().numpy()])

            # Create a dataframe with the predictions
            predictions_df["ticker"] = tickers
            predictions_df["time_idx"] = dates
            predictions_df["actual"] = actuals

            # Merge with original data to get the dates
            predictions_df = predictions_df.merge(
                self.data[["ticker", "time_idx", "date"]],
                on=["ticker", "time_idx"]
            )

            return predictions_df
        else:
            return predictions

    def plot_predictions(self, ticker: str):
        """
        Plot predictions for a specific ticker

        Args:
            ticker: Ticker symbol to plot predictions for
        """
        predictions_df = self.make_predictions(return_x=True)

        # Filter for the specified ticker
        ticker_data = predictions_df[predictions_df["ticker"] == ticker]

        # Sort by date
        ticker_data = ticker_data.sort_values("date")

        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(ticker_data["date"], ticker_data["actual"], label="Actual", color="blue")
        plt.plot(ticker_data["date"], ticker_data["Quantile_0.5"], label="Predicted (Median)", color="red")
        plt.fill_between(
            ticker_data["date"],
            ticker_data["Quantile_0.1"],
            ticker_data["Quantile_0.9"],
            alpha=0.3,
            color="red",
            label="80% Prediction Interval"
        )
        plt.title(f"Stock Price Prediction for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def forecast_future(self, ticker: str, days: int = 30):
        """
        Forecast future prices for a specific ticker

        Args:
            ticker: Ticker symbol to forecast for
            days: Number of days to forecast

        Returns:
            Forecast dataframe
        """
        # Get the latest data for the ticker
        ticker_data = self.data[self.data["ticker"] == ticker].sort_values("time_idx")

        # Get the last time_idx
        last_time_idx = ticker_data["time_idx"].max()

        # Create future dataframe
        future_dates = [ticker_data["date"].max() + timedelta(days=i + 1) for i in range(days)]
        future_time_idx = [last_time_idx + i + 1 for i in range(days)]

        future_df = pd.DataFrame({
            "ticker": [ticker] * days,
            "date": future_dates,
            "time_idx": future_time_idx
        })

        # Add required columns with default values
        for col in self.data.columns:
            if col not in future_df.columns:
                future_df[col] = 0

        # Combine historical and future data
        forecast_data = pd.concat([self.data, future_df]).sort_values(["ticker", "time_idx"])

        # Create forecast dataset
        forecast_dataset = TimeSeriesDataSet.from_dataset(
            self.training, forecast_data, predict=True, stop_randomization=True
        )

        forecast_dataloader = forecast_dataset.to_dataloader(
            batch_size=32, num_workers=0, shuffle=False
        )

        # Make predictions
        predictions = self.model.predict(
            forecast_dataloader,
            return_x=True,
            mode="raw"
        )

        # Extract inputs and predictions
        x, output = predictions
        predictions_df = pd.DataFrame(
            output[0].cpu().numpy(),
            columns=[f"Quantile_{q}" for q in self.model.loss.quantiles]
        )

        # Get the dates and tickers
        dates = x["decoder_time_idx"].cpu().numpy()
        tickers = np.array([self.training.index_to_group_mapping[int(i)] for i in x["groups"].cpu().numpy()])

        # Create a dataframe with the predictions
        predictions_df["ticker"] = tickers
        predictions_df["time_idx"] = dates

        # Merge with forecast data to get the dates
        predictions_df = predictions_df.merge(
            forecast_data[["ticker", "time_idx", "date"]],
            on=["ticker", "time_idx"]
        )

        # Filter for future dates only
        future_predictions = predictions_df[predictions_df["time_idx"] > last_time_idx]

        return future_predictions

    def plot_forecast(self, ticker: str, days: int = 30):
        """
        Plot forecast for a specific ticker

        Args:
            ticker: Ticker symbol to plot forecast for
            days: Number of days to forecast
        """
        # Get historical data
        historical_data = self.data[self.data["ticker"] == ticker].sort_values("date")

        # Get forecast
        forecast_df = self.forecast_future(ticker, days)

        # Plot
        plt.figure(figsize=(14, 7))

        # Plot historical data
        plt.plot(historical_data["date"][-30:], historical_data["Close"][-30:], label="Historical", color="blue")

        # Plot forecast
        plt.plot(forecast_df["date"], forecast_df["Quantile_0.5"], label="Forecast (Median)", color="red")
        plt.fill_between(
            forecast_df["date"],
            forecast_df["Quantile_0.1"],
            forecast_df["Quantile_0.9"],
            alpha=0.3,
            color="red",
            label="80% Prediction Interval"
        )

        plt.title(f"Stock Price Forecast for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_pipeline(self):
        """
        Run the entire pipeline: fetch data, prepare datasets, create model, train, make predictions
        """
        self.fetch_data()
        self.prepare_datasets()
        self.create_model()
        self.train_model()
        return self.make_predictions(return_x=True)


# Example usage
if __name__ == "__main__":
    # Define parameters
    # ticker_symbols = ["AAPL", "AA", "AEP", "FRME"]
    # start_date = "2018-01-01"
    # end_date = "2023-01-01"
    forecast_horizon = 5
    context_length = 30

    # Initialize and run the stock price predictor
    predictor = StockPricePredictor(
        # ticker_symbols=ticker_symbols,
        # # start_date=start_date,
        # # end_date=end_date,
        forecast_horizon=forecast_horizon,
        context_length=context_length
    )

    # Run the pipeline
    predictions = predictor.run_pipeline()

    # Plot predictions for a specific ticker
    predictor.plot_predictions("AAPL")

    # Plot forecast for a specific ticker
    predictor.plot_forecast("AAPL", days=30)
