from typing import Dict

from .basebot import BaseBot , tradeOrder
from .simplelstbot import SimpleBot
import pandas as pd
import numpy as np


class MACrossBot(SimpleBot):
    def __init__(self, name: str = 'macross', params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'stop_loss_pct' : 0.03,  # Stop loss percentage
                            'take_profit_pct': 0.06,  # Take profit percentage
                            'ma_fast':5,
                            'ma_slow':50,
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
        matplotlib.use('Qt5Agg')
        import pylab as plt
        # normalize
        stock_df['price'] = stock_df['price'].values / stock_df['price'].values[0] * 100
        features = self.get_features(
            stock_df)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].plot(stock_df.price.values, label='price')
        axes[0].plot(features['ma_fast'].values, label='ma_fast')
        axes[0].plot(features['ma_slow'].values, label='ma_slow')
        axes[0].legend()
        axes[0].set_title(f' {stock_name}')

        sell_points = np.where([t.order_type == 'sell' for t in trade_signal])[0]
        axes[0].scatter(sell_points, stock_df.price.values[sell_points], s=80, facecolors='none', edgecolors='r',
                        label='sell')
        buy_points = np.where([t.order_type == 'buy' for t in trade_signal])[0]
        axes[0].scatter(buy_points, stock_df.price.values[buy_points], s=80, facecolors='none', edgecolors='b',
                        label='buy')
        axes[0].legend()
        axes[0].set_title(f' {stock_name}')

        axes[1].plot(trade_value_for_this_stock, label='trade with this stock')
        axes[1].plot(reference_index.price.values, label='reference index')
        axes[1].legend()
        return fig


    def get_features(self, stock_df: pd.DataFrame) -> Dict:
        '''
        Get "features" from stocks
        :param stock_df:
        :return:
        '''

        features = {}
        # Moving averages
        features['ma_slow'] = stock_df['price'].rolling(window=self._params['ma_slow']).mean()
        features['ma_fast'] = stock_df['price'].rolling(window=self._params['ma_fast']).mean()


        fast_higher_than_slow = np.zeros(len(features['ma_fast']),)
        fast_higher_than_slow[self._params['ma_fast']:] = \
            (features['ma_fast'][self._params['ma_fast']:] > features['ma_slow'][self._params['ma_fast']:]).astype(int)

        # Set the buy indication
        features['Position'] = np.zeros(len(features['ma_fast']), )
        features['Position'][1:] = np.diff(fast_higher_than_slow)
        #features['Position'] = np.diff(fast_higher_than_slow) wrong

        return features





    def strategy(self, stock_df: pd.DataFrame)->np.array:
        '''


        :param data:
        :return:
        '''

        features  = self.get_features(stock_df)


        trade_signal = np.full(len(stock_df), tradeOrder('hold'))
        startwin = self._params['ma_fast']
        # Generate signals
        nstocks = 0
        for i in range(startwin, len(stock_df)-1):
            if (features['Position'][i]  == 1) and (nstocks == 0) :
                        trade_signal[i] = tradeOrder('buy')
                        nstocks = 100
            elif (features['Position'][i]  == -1) and (nstocks != 0):
                        trade_signal[i] = tradeOrder('sell')
                        nstocks = 0

        return trade_signal







class MACrossV1Bot(MACrossBot):
    def __init__(self, name: str = 'macrossv1', params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'stop_loss_pct' : 0.03,  # Stop loss percentage
                            'take_profit_pct': 0.06,  # Take profit percentage
                            'ma_fast':5,
                            'ma_slow':50,
                            }

    def get_features(self, stock_df: pd.DataFrame) -> Dict:
        '''
        Get "features" from stocks
        :param stock_df:
        :return:
        '''

        features = {}
        # Moving averages
        features['ma_slow'] = stock_df['price'].rolling(window=self._params['ma_slow']).mean()
        features['ma_fast'] =  stock_df['price'].ewm(span=self._params['ma_fast'], adjust=False).mean()

        fast_higher_than_slow = np.zeros(len(features['ma_fast']),)
        fast_higher_than_slow[self._params['ma_fast']:] = \
            (features['ma_fast'][self._params['ma_fast']:] > features['ma_slow'][self._params['ma_fast']:]).astype(int)

        # Set the buy indication
        features['Position'] = np.zeros(len(features['ma_fast']), )
        features['Position'][1:] = np.diff(fast_higher_than_slow)

        return features



class MACrossV2Bot(MACrossBot):
    def __init__(self, name: str = 'macrossv2', params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'stop_loss_pct' : 0.03,  # Stop loss percentage
                            'take_profit_pct': 0.06,  # Take profit percentage
                            'ma_fast':3,
                            'ma_slow':50,
                            }


    def get_features(self, stock_df: pd.DataFrame) -> Dict:
        '''
        Get "features" from stocks
        :param stock_df:
        :return:
        '''

        features = {}
        # Moving averages
        features['ma_slow'] = stock_df['price'].rolling(window=self._params['ma_slow']).mean()
        features['ma_fast'] =  stock_df['price'].ewm(span=self._params['ma_fast'], adjust=False).mean()

        fast_higher_than_slow = np.zeros(len(features['ma_fast']),)
        fast_higher_than_slow[self._params['ma_fast']:] = \
            (features['ma_fast'][self._params['ma_fast']:] > features['ma_slow'][self._params['ma_fast']:]).astype(int)

        # Set the buy indication
        features['Position'] = np.zeros(len(features['ma_fast']), )
        features['Position'][1:] = np.diff(fast_higher_than_slow)

        return features

class MACrossV3Bot(MACrossBot):
    def __init__(self, name: str = 'macrossv3', params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'stop_loss_pct' : 0.03,  # Stop loss percentage
                            'take_profit_pct': 0.06,  # Take profit percentage
                            'ma_fast':3,
                            'ma_slow':50,
                            }


    def get_features(self, stock_df: pd.DataFrame) -> Dict:
        '''
        Get "features" from stocks
        :param stock_df:
        :return:
        '''

        features = {}
        # Moving averages
        features['ma_slow'] = stock_df['price'].rolling(window=self._params['ma_slow']).mean()
        features['ma_fast'] =  stock_df['price']

        fast_higher_than_slow = np.zeros(len(features['ma_fast']),)
        fast_higher_than_slow[self._params['ma_fast']:] = \
            (features['ma_fast'][self._params['ma_fast']:] > features['ma_slow'][self._params['ma_fast']:]).astype(int)

        # Set the buy indication
        features['Position'] = np.zeros(len(features['ma_fast']), )
        features['Position'][1:] = np.diff(fast_higher_than_slow)

        return features