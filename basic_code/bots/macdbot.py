from typing import Dict

from .basebot import BaseBot , tradeOrder
from .simplelstbot import SimpleBot
import pandas as pd
import numpy as np


class macdBot(SimpleBot):
    def __init__(self, name: str = 'macdb', params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'stop_loss_pct' : 0.03,  # Stop loss percentage
                            'take_profit_pct': 0.06,  # Take profit percentage
                            'ma_fast':25,
                            'ma_slow':50,
                            'macd_filt': 9,
                            }


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
        #matplotlib.use('Qt5Agg')
        import pylab as plt
        # normalize
        stock_df['close'] = stock_df['close'].values / stock_df['close'].values[0] * 100
        features = self.get_features(
            stock_df)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].plot(stock_df.close.values, label='close')
        axes[0].plot(features['ma_fast'].values, label='ma_fast')
        axes[0].plot(features['ma_slow'].values, label='ma_slow')
        axes[0].plot(features['signal_line'], label='signal_line')
        axes[0].plot(features['macd_line'], label='macd_line')
        axes[0].plot(features['macd_hist'], label='macd_hist')
        axes[0].legend()
        axes[0].set_title(f' {stock_name}')

        sell_points = np.where([t.order_type == 'sell' for t in trade_signal])[0]
        axes[0].scatter(sell_points, stock_df.close.values[sell_points], s=80, facecolors='none', edgecolors='r',
                        label='sell')
        buy_points = np.where([t.order_type == 'buy' for t in trade_signal])[0]
        axes[0].scatter(buy_points, stock_df.close.values[buy_points], s=80, facecolors='none', edgecolors='b',
                        label='buy')
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
        features['ma_slow'] = stock_df['close'].rolling(window=self._params['ma_slow']).mean()
        features['ma_fast'] = stock_df['close'].rolling(window=self._params['ma_fast']).mean()

        macd_line =  features['ma_fast'] -  features['ma_slow']

        # LP Filtering
        signal_line = macd_line.ewm(span=self._params['macd_filt'], adjust=False).mean()

        # Calculate histogram
        macd_diff = macd_line - signal_line



        features['macd_hist'] = macd_diff
        features['signal_line'] = signal_line
        features['macd_line'] = macd_line

        return features





    def strategy(self, stock_df: pd.DataFrame)->np.array:
        '''


        :param data:
        :return:
        '''

        features  = self.get_features(stock_df)


        trade_signal = np.full(len(stock_df), tradeOrder('hold'))
        startwin = self._params['ma_slow']
        # Generate signals
        nstocks = 0
        for i in range(startwin, len(stock_df)):
            # Buy signal: RSI below oversold + MACD crosses above signal line


            if (features['macd_hist'][i - 1] < 0 and
                    features['macd_hist'][i] > 0) and (nstocks == 0) :
                        trade_signal[i] = tradeOrder('buy')
                        nstocks = 100
            # Sell signal: RSI above overbought + MACD crosses below signal line
            elif ( features['macd_hist'][i - 1] > 0 and
                  features['macd_hist'][i] < 0) and (nstocks != 0):
                        trade_signal[i] = tradeOrder('sell')
                        nstocks = 0


        return trade_signal



class macdWithRSIBot(SimpleBot):
    def __init__(self, name: str = 'macdb_with_rsi', params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'rsi_period': 14,
                            'rsi_overbought' : 30, #should be 70
                            'rsi_oversold': 70, #should be 30
                            'stop_loss_pct' : 0.03,  # Stop loss percentage
                            'take_profit_pct': 0.06,  # Take profit percentage
                            'ma_fast':25,
                            'ma_slow':50,
                            'macd_filt': 9,
                            }


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
        #matplotlib.use('Qt5Agg')
        import pylab as plt
        # normalize
        stock_df['close'] = stock_df['close'].values / stock_df['close'].values[0] * 100
        features = self.get_features(
            stock_df)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].plot(stock_df.close.values, label='close')
        axes[0].plot(features['ma_fast'].values, label='ma_fast')
        axes[0].plot(features['ma_slow'].values, label='ma_slow')
        axes[0].plot(features['rsi'], label='rsi')
        axes[0].plot(features['signal_line'], label='signal_line')
        axes[0].plot(features['macd_line'], label='macd_line')
        axes[0].plot(features['macd_hist'], label='macd_hist')
        axes[0].legend()
        axes[0].set_title(f' {stock_name}')

        sell_points = np.where([t.order_type == 'sell' for t in trade_signal])[0]
        axes[0].scatter(sell_points, stock_df.close.values[sell_points], s=80, facecolors='none', edgecolors='r',
                        label='sell')
        buy_points = np.where([t.order_type == 'buy' for t in trade_signal])[0]
        axes[0].scatter(buy_points, stock_df.close.values[buy_points], s=80, facecolors='none', edgecolors='b',
                        label='buy')
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
        features['ma_slow'] = stock_df['close'].rolling(window=self._params['ma_slow']).mean()
        features['ma_fast'] = stock_df['close'].rolling(window=self._params['ma_fast']).mean()

        macd_line =  features['ma_fast'] -  features['ma_slow']

        # LP Filtering
        signal_line = macd_line.ewm(span=self._params['macd_filt'], adjust=False).mean()

        # Calculate histogram
        macd_diff = macd_line - signal_line



        features['rsi'] = self.calculate_rsi(stock_df['close'].values)
        features['macd_hist'] = macd_diff
        features['signal_line'] = signal_line
        features['macd_line'] = macd_line

        return features





    def strategy(self, stock_df: pd.DataFrame)->np.array:
        '''


        :param data:
        :return:
        '''

        features  = self.get_features(stock_df)

        nstocks = 0
        trade_signal = np.full(len(stock_df), tradeOrder('hold'))
        startwin = self._params['ma_slow']
        # Generate signals
        nstocks = 0
        for i in range(startwin, len(stock_df)):
            # Buy signal: RSI below oversold + MACD crosses above signal line


            if (features['rsi'][i - 1] <  self._params['rsi_oversold'] and
                    features['macd_hist'][i - 1] < 0 and
                    features['macd_hist'][i] > 0) and (nstocks == 0) :
                        trade_signal[i] = tradeOrder('buy')
                        nstocks = 100
            # Sell signal: RSI above overbought + MACD crosses below signal line
            elif (features['rsi'][i - 1] > self._params['rsi_overbought'] and
                  features['macd_hist'][i - 1] > 0 and
                  features['macd_hist'][i] < 0) and (nstocks != 0):
                        trade_signal[i] = tradeOrder('sell')
                        nstocks = 0


        return trade_signal



