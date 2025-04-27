from typing import Dict
from .basebot import BaseBot , tradeOrder
import pandas as pd
import numpy as np
import ta
import pylab as plt

class DefaultBot(BaseBot):
    def __init__(self, name: str = 'default' , params : Dict = None):
        self._name = name
        self._params = params

    def strategy(self, data: pd.DataFrame)->np.array:
        '''

        :param data:
        :return:
        '''
        # Don't buy anything - use alternative
        trade_signal = np.full(len(data), tradeOrder('sell'))

        return trade_signal
    def display(self, stock_name : str, stock_df: pd.DataFrame ,
                trade_signal : np.array, reference_index : pd.DataFrame = None)->"fig":
        fig = plt.figure()
        plt.plot(stock_df.close.values, label='close')

        return fig


class SimpleBot(BaseBot):
    def __init__(self, name: str = 'simple', params: Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {
            }

    def atr_trailing_stop(df, atr_period=14, multiplier=3):
        """
        Adds ATR trailing stop loss levels to the dataframe.
        """
        # Calculate ATR
        df['ATR'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'],
                                                     window=atr_period)

        # Initialize trailing stop columns
        df['Trailing_Stop_Long'] = df['Close'] - multiplier * df['ATR']
        df['Trailing_Stop_Short'] = df['Close'] + multiplier * df['ATR']

        return df

    def find_support(self , prices_low, window):
        '''
        Find support levels
        :param prices_low:
        :param window:
        :return:
        '''
        support_levels = []
        for i in range(window, len(prices_low)):
            low = prices_low.iloc[i - window:i]
            if prices_low.iloc[i] < low.min():
                support_levels.append((prices_low.index[i], prices_low.iloc[i]))
        return support_levels

    def calculate_rsi(self , prices, period=14):
        """
        Calculate the Relative Strength Index (RSI) for a given price series.

        Parameters:
        -----------
        prices : array-like
            Price series data (typically closing prices)
        period : int, optional (default=14)
            The lookback period for RSI calculation

        Returns:
        --------
        array-like
            RSI values for the given price series
        """
        # Calculate price changes
        deltas = np.diff(prices)

        # Create arrays for gains and losses
        gains = np.copy(deltas)
        losses = np.copy(deltas)

        # Separate gains and losses
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        # First values based on simple average
        if len(gains) >= period:
            avg_gain[period] = np.mean(gains[:period])
            avg_loss[period] = np.mean(losses[:period])

        # Calculate subsequent values with smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        # Calculate RS and RSI
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)

        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))

        return rsi

    def calculate_cci(self, high, low, close, period=20):
        """
        Calculate the Commodity Channel Index (CCI) using high, low, and close prices.

        Parameters:
        -----------
        high : array-like
            Series of high prices
        low : array-like
            Series of low prices
        close : array-like
            Series of closing prices
        period : int, optional (default=20)
            The lookback period for CCI calculation

        Returns:
        --------
        array-like
            CCI values for the given price data
        """
        # Calculate typical price (TP)
        tp = (high + low + close) / 3

        # Initialize arrays
        sma_tp = np.zeros_like(tp)
        mad = np.zeros_like(tp)
        cci = np.zeros_like(tp)

        # Calculate SMA of typical price and Mean Deviation
        for i in range(period - 1, len(tp)):
            sma_tp[i] = np.mean(tp[i - period + 1:i + 1])
            mad[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - sma_tp[i]))

            # Calculate CCI
            if mad[i] > 0:
                cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mad[i])

        return cci

    def display(self, stock_name: str, stock_df: pd.DataFrame,
                trade_signal: np.array, reference_index: pd.DataFrame = None,
                trade_value_for_this_stock: np.array = None) -> "fig":
        '''

        :param stock_name:
        :param stock_df:
        :param trade_signal:
        :param reference_index:
        :return:
        '''
        import matplotlib
        import pylab as plt
        # normalize
        stock_df['close'] = stock_df['close'].values / stock_df['close'].values[0] * 100
        features = self.get_features(
            stock_df)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].plot(stock_df.close.values, label='close')
        axes[0].plot(features['ma_50'].values, label='ma_50')
        axes[0].plot(features['ma_150'].values, label='ma_150')
        axes[0].plot(features['ma_200'].values, label='ma_200')
        axes[0].plot(features['rsi'], label='rsi')
        axes[0].plot(features['cci'], label='cci')
        axes[0].legend()
        axes[0].set_title(f' {stock_name}')

        axes[1].plot(trade_value_for_this_stock, label='trade with this stock')
        axes[1].plot(reference_index.close.values, label='reference index')
        axes[1].legend()
        return fig

    def get_features(self, stock_df: pd.DataFrame) -> Dict:
        '''
        Get "features" from stocks
        :param stock_df:
        :param do_normalize:
        :return:
        '''

        features = {}
        # Moving averages
        features['ma_200'] = stock_df['close'].rolling(window=200).mean()
        features['ma_150'] = stock_df['close'].rolling(window=150).mean()
        features['ma_50'] = stock_df['close'].rolling(window=50).mean()

        features['rsi'] = self.calculate_rsi( stock_df['close'].values)
        features['cci'] = self.calculate_cci(stock_df['2. high'].values ,stock_df['3. low'].values,  stock_df['close'].values )

        return features

    def strategy(self, stock_df: pd.DataFrame) -> np.array:
        '''


        :param data:
        :return:
        '''

        features = self.get_features(stock_df)


        trade_signal = np.full(len(stock_df), tradeOrder('hold'))

        return trade_signal



