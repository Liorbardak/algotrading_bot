import pandas as pd
import numpy as np
from config.config import ConfigManager
from utils.protofolio import Portfolio

class TradingPolicy:
    _registry = {}
    @classmethod
    def register(cls, name):
        def decorator(policy_class):
            cls._registry[name] = policy_class
            return policy_class

        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown policy: {name}")
        return cls._registry[name](*args, **kwargs)

@TradingPolicy.register("MostBasic")
class TradingPolicyMostBasic(TradingPolicy):
    def __init__(self ,config,
                 start_date = None , end_date = None):
        self.name = "MostBasic"
        self.config = config
        self.portfolio = Portfolio()
        self.start_date  = start_date
        self.end_date = end_date

    def score_ticker_to_buy(self, date, ticker, tickers_df, complement_df, portfolio_weight):
        '''
        Should we buy this ticker at this date ?
        Score this option using current day / historical data
        '''

        # Init ticker score at minimal level
        ticker_score = {'weighted_score': 0 , 'price_based_score': 0 ,  'complement_based_score': 0,
                              'portfolio_based_score':0 }

        #####################################
        # portfolio based score
        #####################################
        # Check portion of this ticker in the portfolio
        if portfolio_weight > self.config.get_parameter('portfolio','max_portion_per_ticker'):
            # portfolio portion is too high - no need check other
            return ticker_score
        else:
            # for now  , pass/fail  score TODO - revisit
            ticker_score['portfolio_based_score'] = 100

        #####################################
        # Compliments based score
        #####################################
        # Get complement before current date
        complements_before_current_time = complement_df[(complement_df.ticker == ticker) & (
                    complement_df.Date.values.astype('datetime64[D]') <= np.datetime64(date))]
        if (len(complements_before_current_time) == 0):
            # no complements received  yet - do not buy
            return ticker_score

        # Look at last complements
        last_complements = complements_before_current_time[
            complements_before_current_time.Date == complements_before_current_time.Date.max()]
        last_complements_Date = pd.Timestamp(last_complements.Date.values[0], tz='UTC')

        if (date - last_complements_Date) > pd.Timedelta(days=90):
            # last complements received  more than quarter ago - don't count on it - dont buy  TODO - revisit
            return ticker_score


        # Compliments condition to buy

        number_of_analysts_comp = last_complements.number_of_analysts_comp.values[0]
        total_number_of_analysts = last_complements.total_number_of_analysts.values[0]

        # Rename
        min_complements_th1 =   self.config.get_parameter('buy', 'min_complements_th1')
        min_complements_th2 =   self.config.get_parameter('buy', 'min_complements_th2')
        complements_portion_th2 = self.config.get_parameter('buy', 'complements_portion_th2')

        ratio_of_complementers = number_of_analysts_comp / (total_number_of_analysts + 1e-6) > complements_portion_th2

        compliments_buy_indication = (((number_of_analysts_comp >= min_complements_th2) & ratio_of_complementers)
                                     |
                                      (number_of_analysts_comp  >= min_complements_th1))

        if ~compliments_buy_indication:
            # not good enough
            return ticker_score
        else:
            # for now  , only pass/fail  score TODO - revisit
            ticker_score['complement_based_score'] = 100

        #####################################
        # Price based score
        #####################################

        # Complements are good enough for buying - check stock prices
        ticker_df = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date <= date)]
        ma_slop_is_positive = ticker_df.ma_150_slop.values[-1] > 0
        if (date - last_complements_Date) < pd.Timedelta(days=2):
            # 1-2 Days after the complements -  price is above ma150
            price_above_ma150 = ticker_df.ma_150.values[-1] < ticker_df.Close.values[-1]
        else:
            # 1-2 Days after the complements -  10 days price is above ma150
            price_above_ma150 = np.all(ticker_df.ma_150.values[-10:] < ticker_df.Close.values[-10:])

        # price condition to buy
        if price_above_ma150 & ma_slop_is_positive:
            # for now  , only pass/fail  score TODO - revisit
            ticker_score['price_based_score'] = 100

        # Calculate overall score - naive TODO - revisit
        ticker_score['weighted_score']  = np.min([ticker_score['portfolio_based_score'],
                                                  ticker_score['price_based_score'],
                                                  ticker_score['complement_based_score']])
        return ticker_score

    def buy(self , date , tickers_score):
        '''
        Should we buy this ticker at this date ?
        Choose the potion to buy for each of the new ticker
        May also choose to sell
        '''

    def sell(self,  date , ticker , tickers_df , complement_df ,  portfolio_weight):
        '''
        Should we sell this ticker at this date?
        '''
        if  self.portfolio.is_in(ticker):
            # Not in portfolio - nothing to sell from this stock
            return

        ma_200 = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date == date)].ma_200.values[0]
        current_price = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date == date)].Close.values[0]
        if ~np.isnan(ma_200) and ma_200 > current_price:
            # Sell all and return
            return

        # Check portion of this ticker in the portfolio
        if portfolio_weight > self.config.get_parameter('portfolio','max_portion_per_ticker'):
            # portfolio portion is too high - partial sell
            return

    def trade_recurrent(self , date , tickers , tickers_df , complement_df , reference_index):
        '''
        Manage the portfolio - decide what to buy /sell at this date
        :param tickers: all tickers that can be considered for trade
        :return:
        '''
        portfolio_weights = self.portfolio.get_portfolio_weights()

        #################
        # sell tickers
        #################
        for ticker in tickers:
            portfolio_weight = portfolio_weights[ticker] if ticker in portfolio_weights.keys() else 0
            self.sell(date , ticker , tickers_df , complement_df , portfolio_weight)

        #################
        # buy tickers
        #################

        # score tickers
        portfolio_weights = self.portfolio.get_portfolio_weights()
        tickers_score = dict()
        for ticker in tickers:
            portfolio_weight = portfolio_weights[ticker] if ticker in portfolio_weights.keys() else 0
            tickers_score['ticker'] = self.score_ticker_to_buy(date , ticker , tickers_df , complement_df, portfolio_weight)

        # buy tickers
        self.buy(date , tickers_score)

        #################
        # buy reference index if there is residual cash
        #################





    def trade(self , tickers_df , complement_df , reference_index ):
        '''
        Trade simulation
        '''

        # Simplistic simulation
        # - simulation per ticker
        # - price gain/loss only on times were we actually bought the stock
        for ticker, tdf in tickers_df.groupby('ticker'):
            self.portfolio = Portfolio()
            for date in tdf.Date:
                print(date)
                self.trade_recurrent( date, [ticker] ,  tickers_df , complement_df , reference_index )



if __name__ == "__main__":

    policy = TradingPolicy.create("MostBasic", config= ConfigManager(), start_date= 4, end_date = 1)
    print(policy.name)
    print(policy.start_date)
