import os

import pandas as pd
import numpy as np
from config.config import ConfigManager
from utils.protofolio import Portfolio
from copy import copy
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
    def __init__(self ,config, default_index_name = 'snp' ):
        self.name = "MostBasic"
        self.config = config
        self.portfolio = Portfolio()
        self.default_index_name = default_index_name

    def score_tickers(self, date, tickers_df, complement_df):

        # Score tickers
        portfolio_weights = self.portfolio.get_portfolio_weights()
        tickers_score = dict()
        # Re-score tickers already in the portfolio
        for ticker, portfolio_weight  in portfolio_weights.items():
            score = self.score_ticker(date, ticker, tickers_df, complement_df, portfolio_weight)

            if score['weighted_score'] > 0:
                # Store only tickers with positive score - valid for buying
                tickers_score[ticker] = score

        # Score tickers  not in the portfolio
        tickers_not_in_portfolio = set(complement_df[complement_df.Date.dt.normalize() == date].ticker) - set(portfolio_weights.keys())
        for ticker in tickers_not_in_portfolio:
            score = self.score_ticker(date, ticker, tickers_df, complement_df, 0)
            if score['weighted_score'] > 0:
                # Store only tickers with positive score - valid for buying
                tickers_score[ticker] = score
        return tickers_score

    def score_ticker(self, date, ticker, tickers_df, complement_df, portfolio_weight):
        '''
        Should we buy this ticker at this date ?
        Score this option using current day / historical data
        '''

        # Init ticker score at minimal level
        ticker_score = {'weighted_score': 0 , 'price_based_score': 0 ,  'complement_based_score': 0,
                              'portfolio_based_score':0 ,'portfolio_weight': portfolio_weight }

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

    def buy(self , date , tickers_score , tickers_df , default_index_df):
        '''
        Should we buy this ticker at this date ?
        Choose the potion to buy for each of the new ticker
        May also choose to sell
        '''

        number_of_stocks_in_portfolio = len(self.portfolio.positions)
        # Check if there are new stocks  to add
        new_tickers = list(set(tickers_score.keys()) - set(self.portfolio.positions.keys()))

        number_of_new_new_tickers_to_add = np.min(
            [self.config.get_parameter('portfolio', 'max_number_of_tickers') - number_of_stocks_in_portfolio,
             len(new_tickers)])

        ####################################
        #   Add new tickers to the portfolio
        ####################################

        current_portfolio_weights = self.portfolio.get_portfolio_weights()
        new_portfolio_weights = copy(current_portfolio_weights)

        if number_of_new_new_tickers_to_add > 0:

            # Set new tickers to add - TODO - choose by weighted score
            new_tickers_to_add = new_tickers[:number_of_new_new_tickers_to_add]


            if number_of_stocks_in_portfolio:
                # Set the weight of the new tickers as the average weight in the portfolio

                average_ticker_weight = sum(current_portfolio_weights.values()) / len(current_portfolio_weights)
                new_ticker_weight = np.min([average_ticker_weight , self.config.get_parameter('portfolio', 'max_portion_per_ticker')])

                # add the new tickers
                new_portfolio_weights.update({ticker: new_ticker_weight for ticker in new_tickers_to_add})
            else:
                new_ticker_weight = self.config.get_parameter('portfolio', 'max_portion_per_ticker')
                new_portfolio_weights = {ticker: new_ticker_weight for ticker in new_tickers_to_add}

        # Clip weight to the maximal allow
        for ticker in new_portfolio_weights.keys():
            new_portfolio_weights[ticker] = np.min([new_portfolio_weights[ticker] , self.config.get_parameter('portfolio', 'max_portion_per_ticker')])


        total_weights = sum(new_portfolio_weights.values())

        if total_weights > 1.0:
            # normalize weights - total_weights should do not exceed
            new_portfolio_weights =  {ticker: weight / total_weights  for ticker,weight  in new_portfolio_weights.items()}
        elif total_weights < 1.0:
            # There is free cash - add it to stocks that does not
            free_cash_weight = total_weights - 1.0
            # Add the free chash equally  to all tickers


        ###############################################
        # Sell / buy tickers with the new distribution
        ###############################################
        total_value = self.portfolio.get_total_value()
        for ticker, new_ticker_weight in new_portfolio_weights.items():
            ticker_price = tickers_df[tickers_df.ticker == ticker].Close.values[0]
            if ticker not in current_portfolio_weights:
                # new ticker
                value_per_stock = total_value * new_ticker_weight
                ticker_price = tickers_df[tickers_df.ticker == ticker].Close.values[0]
                quantity = value_per_stock / ticker_price
                self.portfolio.buy_stock(ticker, quantity, ticker_price)
            else:
                # existing ticker - sell or buy
                current_ticker_weight = current_portfolio_weights[ticker]
                value_per_stock = total_value * (new_ticker_weight - current_ticker_weight)

                if value_per_stock > 0:
                    #buy
                    quantity = value_per_stock / ticker_price
                    self.portfolio.buy_stock(ticker, quantity, ticker_price, date)
                else:
                    #sell
                    quantity = -value_per_stock / ticker_price
                    self.portfolio.sell_stock(ticker, quantity, ticker_price , date)

        # verify free cash is not negative
        print(f"portfolio  {date}, {self.portfolio.get_portfolio_weights()}")

        assert self.portfolio.cash + 1e-6 >= 0, f"free cash is negative  {self.portfolio.cash} {date}"

    def sell(self,  date , ticker , tickers_df , complement_df ,  portfolio_weight):
        '''
        Should we sell this ticker at this date?
        '''
        if  self.portfolio.is_in(ticker):
            # Not in portfolio - nothing to sell from this stock
            return

        # Sell condition
        ma_200 = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date == date)].ma_200.values[0]
        current_price = tickers_df[(tickers_df.ticker == ticker) & (tickers_df.Date == date)].Close.values[0]
        sell_condition = ~np.isnan(ma_200) and ma_200 > current_price

        if sell_condition:
            # Sell all holdings for this ticker
            ticker_price = tickers_df[tickers_df.ticker == ticker].Close.values[0]
            self.portfolio.sell_stock(ticker,  self.portfolio.positions[ticker].quantity, ticker_price, date)



    def trade_recurrent(self , date , tickers , tickers_df , complement_df , default_index):
        '''
        Manage the portfolio - decide what to buy /sell at this date
        :param tickers: all tickers that can be considered for trade
        :return:
        '''

        # Update existing stocks in the portfolio
        self.portfolio.update_prices(tickers_df[tickers_df.Date == date] , default_index[default_index.Date == date] , date)

        portfolio_weights = self.portfolio.get_portfolio_weights()

        ######################################################################################################
        # "sell" all  reference index - get cash for buying
        #  will convert the unused  cash back to the reference index at the end of this day
        ######################################################################################################
        self.portfolio.sell_all_default_index(default_index[default_index.Date == date].Close.values[0] , date)

        ###################################################
        # sell tickers that does not meet price criteria
        ###################################################
        for ticker, portfolio_weight in portfolio_weights.items():
            self.sell(date , ticker , tickers_df , complement_df , portfolio_weight)

        #################
        # buy tickers
        #################
        # Score tickers in portfolio
        tickers_score = self.score_tickers( date, tickers_df, complement_df)
        # decide which tickers to buy
        self.buy(date , tickers_score , tickers_df[tickers_df.Date.dt.normalize() == date] , default_index[default_index.Date == date])

        ###################################################
        # buy reference index if there is residual cash
        ###################################################
        
        self.portfolio.buy_default_index_with_all_cash(default_index[default_index.Date == date].Close.values[0] , date)





    def trade(self , tickers_df , complement_df , default_index , outputpath = None , start_date = None , end_date = None ):
        '''
        Trade simulation
        '''
        self.portfolio = Portfolio()
        #
        tickers = list(set(complement_df.ticker))

        # Set running dates
        dates = tickers_df.Date
        dates = np.array(sorted(list(set(dates))))
        if start_date is not None:
            dates = dates[dates >= pd.Timestamp(start_date).tz_localize('UTC')]
        if end_date is not None:
            dates = dates[dates <= pd.Timestamp(end_date).tz_localize('UTC')]

        for date in dates:
            self.trade_recurrent( date, tickers ,  tickers_df , complement_df , default_index )

        # Save results
        if outputpath is not None:
            os.makedirs(outputpath, exist_ok=True)
            outfile = os.path.join(outputpath, 'trade_sim.csv')
            self.portfolio.history_to_scv(outfile)




if __name__ == "__main__":

    policy = TradingPolicy.create("MostBasic", config= ConfigManager())
    print(policy.name)

