import pandas as pd
import numpy as np

class TradingPolicyBase:
    def __init__(self):
        pass

    def can_buy(self ,portfolio, date , ticker , tickers_df , complement_df):
        '''
        Should we buy this ticker at this date ?
        Score this option using current day/ historical data
        '''
        # Get complement before current date
        complements_before_current_time = complement_df[(complement_df.ticker == ticker) & (complement_df.Date.values.astype('datetime64[D]') <= np.datetime64(date))]
        if(len(complements_before_current_time) == 0):
            # no complements received  yet - do not buy
            return {'buy': False,'score' : 0}

        # Look at last complements
        last_complements = complements_before_current_time[complements_before_current_time.Date == complements_before_current_time.Date.max()]

        if (date- last_complements.Date) > pd.Timedelta(days = 90):
            # last complements received  more than quarter ago - don't count on it
            return {'buy': False, 'score': 0}

        # Compliments condition to buy
        number_of_analysts_comp = last_complements.number_of_analysts_comp.values[0]
        total_number_of_analysts = last_complements.total_number_of_analysts.values[0]

        compliments_buy_indication = ((number_of_analysts_comp >= 3) & (
                number_of_analysts_comp / (total_number_of_analysts + 1e-6) > 0.5)) | (number_of_analysts_comp >= 5)

        if ~compliments_buy_indication:
            return {'buy': False, 'score': 0}

        # Complements are good enough for buying - check stock prices
        ticker_df = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date <= date)]
        ma_slop_is_positive = ticker_df.ma_150_slop.values[-1] > 0
        if  (date- last_complements.Date) < pd.Timedelta(days = 2):
            # 1-2 Days after the complements -  price is above ma150
            price_above_ma150 = ticker_df.ma_150.values[-1] < ticker_df.Close.values[-1]
        else:
            # 1-2 Days after the complements -  10 days price is above ma150
            price_above_ma150 = np.all(ticker_df.ma_150.values[-10:] < ticker_df.Close.values[-10:])
        if  price_above_ma150 & ma_slop_is_positive:
            return {'buy': True, 'score': 100}

        return {'buy': False, 'score': 0}
    def can_sell(self, portfolio , date , ticker , tickers_df , complement_df):
        '''
        Should we sell this ticker at this date?
        Score this option using current day / historical data
        '''
        if (not ticker in portfolio.keys()) or portfolio[ticker] == 0:
            # Nothing to sell from this stock
            return {'sell': False, 'score': 0}

        ma_200 = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date == date)].ma_200.values[0]
        current_price = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date == date)].Close.values[0]
        if ~np.isnan(ma_200) and ma_200 > current_price:
            return {'sell': True, 'score': 100}
        else:
            return {'sell': False, 'score': 0}

    def trade_one_day(self ,portfolio,  date , tickers , tickers_df , complement_df , reference_index):
        '''
        Manage the portfolio - decide what to buy /sell at this date
        :param tickers: all tickers that can be considered for trade
        :return:
        '''
        for ticker in tickers:

            sell_command = self.can_sell(portfolio , date , ticker , tickers_df , complement_df)
            buy_command = self.can_buy(portfolio , date , ticker , tickers_df , complement_df)


    def trade(self , dates_range , tickers_df , complement_df , reference_index ):
        '''
        Trade simulation
        '''

        # Simplistic simulation
        # - simulation per ticker
        # - price gain/loss only on times were we actually bought the stock
        for ticker, tdf in tickers_df.groupby('ticker'):

            portfolio = {ticker : 0,'reference_index' : 1 }
            for date in tdf.Date:
                self.trade_one_day(portfolio , date, [ticker] ,  tickers_df , complement_df , reference_index )
