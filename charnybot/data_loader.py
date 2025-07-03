import pandas as pd
import numpy as np
import os
import json
from config.config import TICKERS_DIR, COMPLEMENT_RESULTS_DIR
import datetime
class FinancialDataLoaderBase:
    def __init__(self):
        pass

    def get_stock_features(self, stock_df):
        '''
        Add features to stock financial date
        '''

        stock_df['ma_200'] = stock_df['Close'].rolling(window=200).mean()
        stock_df['ma_150'] = stock_df['Close'].rolling(window=150).mean()
        ma_150_diff = np.diff(stock_df['ma_150'].values)
        stock_df['ma_150_slop'] = np.hstack((ma_150_diff[0], ma_150_diff))

        return stock_df

    def load_stock_data(self, tickers , min_max_dates = None , get_average_stock = False):
        """
        Load historical stock data
        """
        dfs = []
        actual_min_max_dates = None
        for ticker in tickers:
            df = pd.read_csv(os.path.join(TICKERS_DIR,ticker, 'stockPrice.csv'))

            for kl in [k for k in df.keys() if 'Unnamed' in k]:
                df = df.drop(kl, axis=1)

            df['ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'],utc=True)
            if min_max_dates is not None:
                # Take only range in time
                df = df[(df.Date >= pd.to_datetime(min_max_dates[0],utc=True)) & (df.Date <= pd.to_datetime(min_max_dates[1],utc=True))]
            if actual_min_max_dates is None:
                actual_min_max_dates = [df.Date.min(), df.Date.max()]
            else:
                actual_min_max_dates = [np.max([df.Date.min(), actual_min_max_dates[0]]), np.min([df.Date.max(), actual_min_max_dates[1]])]

            df = self.get_stock_features(df)
            dfs.append(df)

        all_df = pd.concat(dfs)
        all_df.reset_index(drop=True, inplace=True)

        # Calculate the average of all stocks
        avg_df = None
        keys_to_avg = ['High', 'Low', 'Open', 'Close', 'AdjClose', 'Volume']
        if get_average_stock:
            # Calculate the average of all stocks
            for ticker, tdf in all_df.groupby('ticker'):
                # Get the stock in the time range
                tdf = tdf[(pd.to_datetime(tdf.Date) >= actual_min_max_dates[0]) & (pd.to_datetime(tdf.Date) <= actual_min_max_dates[1])]
                # Normalize price by the first date
                for k in keys_to_avg:
                    tdf[k] = tdf[k] / tdf.Close.values[0]
                if avg_df is None:
                    avg_df = tdf
                else:
                    for k in keys_to_avg:
                        avg_df[k] = avg_df[k].values + tdf[k].values

            for k in keys_to_avg:
                avg_df[k] = avg_df[k].values / len(tickers)


        return all_df , actual_min_max_dates, avg_df

    def load_complement_data(self, tickers , min_max_dates = None):
        """
        Load historical stock data
        """
        dfs = []
        for ticker in tickers:
            comps = json.load(open(os.path.join(COMPLEMENT_RESULTS_DIR, ticker + '_compliment_summary.json')))
            df = pd.DataFrame(comps)
            df['Date'] = pd.to_datetime(df['date'], utc=True)
            df['ticker'] = ticker
            df['number_of_analysts_comp'] = (df['number_of_analysts_comp_1'] + df['number_of_analysts_comp_2'] +
                                             df['number_of_analysts_comp_3'])
            if min_max_dates is not None:
                # Take only range in time
                df = df[(df.Date >= pd.to_datetime(min_max_dates[0],utc=True)) & (df.Date <= pd.to_datetime(min_max_dates[1],utc=True))]

            dfs.append(df)
        all_df = pd.concat(dfs)
        all_df.reset_index(drop=True, inplace=True)
        return all_df

    def load_all_data(self, tickers , min_max_dates = None , get_average_stock = False):
        '''
        Load historical stock data
        :param tickers:
        :param min_max_dates:
        :return:
        '''
        stocks_df , actual_min_max_dates, avg_df  = self.load_stock_data(tickers, min_max_dates, get_average_stock = get_average_stock)
        complement_df = self.load_complement_data(tickers, min_max_dates)



        return {'stocks_df': stocks_df , 'complement_df': complement_df , 'actual_min_max_dates': actual_min_max_dates ,
                'avg_df' : avg_df }




if __name__ == "__main__":
    fl = FinancialDataLoaderBase()
    print(fl.load_complement_data(['ADM', 'AJG']))

    print(fl.load_stock_data(['ADM' , 'AJG' ] ,min_max_dates = ['2023-01-01', '2025-01-01'] ))
