import pickle
import pandas as pd
import numpy as np
import os
import ta
import shutil
import pylab as plt
from typing import Dict, List
from datetime import datetime
from copy import deepcopy
from ML.Predictor.config.config import get_config



def detect_stocks_with_jumps(inputdir: str, stocks_names: np.array, th=0.7) -> List:
    '''
    Detect stocks with very large jumps in price - omit from training
    '''
    bad_stocks = []
    good_stocks = []
    ngood = 0
    nbad = 0
    for stock_name in stocks_names:

        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        price = np.log(df['close'].values)
        log_price = np.log(np.abs(price) + 1)
        if np.any(np.abs(np.diff(np.diff(np.diff(log_price)))) > th):
            bad_stocks.append(stock_name)
            nbad += 1
            print(f' {stock_name} is bad {ngood} {nbad}')

        else:
            good_stocks.append(stock_name)
            ngood += 1
            print(f' {stock_name} is good {ngood} {nbad}')

        # plt.figure()
        # plt.plot(price)
        # plt.plot(log_price)
        # plt.plot(np.diff(log_price))
        # plt.plot(np.abs(np.diff(np.diff(np.diff(log_price)))))
        # plt.title(stock_name)

    # plt.show()
    return good_stocks, bad_stocks


def re_arange_df(df: pd.DataFrame, ticker: str, stock_id: int, norm_factors: Dict):
    # Add some stuff / normalize / so on

    df['time_idx'] = np.arange(len(df))
    df['year'] = [v.astype('datetime64[Y]').astype(int) + 1970 for v in df.date.values]
    df['month'] = [v.astype('datetime64[M]').astype(int) % 12 + 1 for v in df.date.values]
    df['day'] = [(v - v.astype('datetime64[M]')).astype(int) + 1 for v in df.date.values]
    df['ticker'] = ticker
    df['stock_id'] = stock_id

    df = df[['date', 'year', 'month', 'day', 'open', 'close', 'high', 'low', 'volume', 'ticker', 'stock_id']]







    # Add more features
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()/100
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'] , df['low'] , df['close']).stoch()/100
    df['ma20'] =  df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()

    # Normalize to by scaling (without offset , for now (?))
    normFact =1.0 / df['close'].values[0]
    for k in norm_factors['cols_to_pre_normalize_together']:
        df.loc[:, k] = (df[k].astype(float) * normFact).astype(df[k].dtype)

    return df, normFact


def get_normalized_training_data(inputdir, stock_list_to_use, params : Dict , first_ind = 0, minDate=None, maxDate=None,
                                 min_length: int = 150,ticker_to_id : Dict = None

                                 ):
    '''
    Prepare data for training - sort out, normalize , rename , add features ...
    :param inputdir: directory of the raw ticker data
    :param stock_list_to_use:
    :param minDate:
    :param maxDate:
    :param min_length:
    :return: training data , normalization factors , data without normalization
    '''

    norm_factors = {}
    norm_factors['cols_to_pre_normalize_together'] = params['features_normalize_together'].split(',')

    dfs = []
    dfs_raw = []
    ind = first_ind
    for idx , ticker in enumerate(stock_list_to_use):
        print(f" {ticker} {idx+1} out of {len(stock_list_to_use)}")
        df = pd.read_excel(os.path.join(inputdir, ticker, 'stockPrice.xlsx'), engine='openpyxl')

        # Rename
        df.rename(
            columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', 'close': 'close', '5. volume': 'volume',
                     'Date': 'date'},
            inplace=True)
        # Remove redundant cols
        df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
        if minDate is not None and maxDate is not None:
            min_date = datetime.strptime(minDate, "%Y-%m-%d")
            max_date = datetime.strptime(maxDate, "%Y-%m-%d")
            # take only these dates
            #datetime.strptime(minDate, "%Y-%m-%d")

            df = df[df['date'].between(min_date, max_date)]

        if (len(df) < min_length):
            continue
        # Store non-normalized data
        df_raw = deepcopy(df)
        df_raw['ticker'] = ticker
        dfs_raw.append(df_raw)
        if ticker_to_id is None:
            ticker_id = ind
        else:
            ticker_id = ticker_to_id[ticker]
        df, normFact = re_arange_df(df, ticker, ticker_id, norm_factors)

        ind = ind+1

        dfs.append(df)
        norm_factors[ticker + 'normFact'] = normFact

    # Get the raw (non-normalized) data
    raw_df = pd.concat(dfs_raw, ignore_index=True)

    train_df = pd.concat(dfs, ignore_index=True)
    # Fill NA values
    train_df = train_df.fillna(0)

    # Create time index - ensure each ticker starts from 0
    train_df = train_df.sort_values(['ticker', 'date'])
    train_df['time_idx'] = train_df.groupby('ticker').cumcount()

    # Drop rows with missing values
    train_df = train_df.dropna()

    # Rescale volume and other features that needs scaling on all data
    cols_to_normalize = ['volume']
    for k in cols_to_normalize:
        norm_factors[k + 'offset'] = np.min(train_df['volume'].values)
        norm_factors[k + 'scale'] = 1.0 / (np.max(train_df['volume'].values) - norm_factors[k + 'offset'])
        train_df[k] = (train_df[k] - norm_factors[k + 'offset']) * norm_factors[k + 'scale']

    return train_df, norm_factors, raw_df


def get_good_stocks_out_of_snp(recalc = False):
    '''
     get "good" stocks to train out of the snp
     :return:
    '''

    if recalc:
        # add more "good" stocks
        # Get "known " good stocks
        good_names = pickle.load(open('C:/Users/dadab/projects/algotrading/data/training/goodstocks_v1.pkl', 'rb'))
        inputdir = 'C:/Users/dadab/projects/algotrading/data/tickers'

        snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/snp500/all_stocks.csv')

        # go over all the existing stocks , see of they can be added
        all_stocks = np.array([d for d in os.listdir(inputdir) if
                               (os.path.isdir(os.path.join(inputdir, d)) )])


        stocks_to_check = list(set(all_stocks) - set(snp['ticker'])-set(good_names))


        # Get the bad stocks
        good_stocks, bad_stocks = detect_stocks_with_jumps(inputdir, stocks_to_check,th = 0.3)

        good_stocks = list(set(good_names) | set(good_stocks))

        pickle.dump(good_stocks, open('C:/Users/dadab/projects/algotrading/data/training/goodstocks_v2_large.pkl', 'wb'))
    else:
        # return the precalcuated "good" stocks
        good_stocks = pickle.load(open('C:/Users/dadab/projects/algotrading/data/training/goodstocks_v2_large.pkl', 'rb'))

    return good_stocks



def create_set(inputdir: str, outputdir: str,params : Dict, all_training_stocks: np.array, all_test_stocks: np.array, val_train_split,
               start_train_date, end_train_date, start_test_date, end_test_date , use_ma = False , overfit = False):
    '''
    Create training set
    :param inputdir:
    :param outputdir:
    :param all_stocks:
    :param val_train_split:
    :param start_train_date:
    :param end_train_date:
    :param start_test_date:
    :param end_test_date:
    :return:
    '''

    os.makedirs(outputdir, exist_ok=True)


    in_train_but_not_in_test = np.array(list(set(all_training_stocks)- set(all_test_stocks)))
    in_train_and_in_test = all_test_stocks

    if overfit:
        # overfit
        in_train_and_in_test = in_train_and_in_test[:3]
        in_train_but_not_in_test= in_train_but_not_in_test[:11]



    # Get all data
    all_df, all_norm_factors, all_df_orig = get_normalized_training_data(inputdir, in_train_and_in_test,params=params)


    # Get  test set on later times
    min_date = datetime.strptime(start_test_date, "%Y-%m-%d")
    max_date = datetime.strptime(end_test_date, "%Y-%m-%d")
    test_df = all_df[all_df['date'].between(min_date, max_date)]
    test_df_orig = all_df_orig[all_df_orig['date'].between(min_date, max_date)]


    # Add extra stocks - only to the train and validation sets
    if(len(in_train_but_not_in_test) > 10):
        # get  extra data for training

        extra_df, extra_norm_factors, _ = get_normalized_training_data(inputdir, in_train_but_not_in_test,
                                                                            params=params,
                                                                            first_ind=all_df.stock_id.max() + 1,
                                                                           minDate = start_train_date,
                                                                           maxDate = end_train_date
                                                                           )
        # merge
        all_df = pd.concat([all_df, extra_df]).reset_index(drop=True)
        all_norm_factors.update(extra_norm_factors)

    # Create train & validation sets
    min_date = datetime.strptime(start_train_date, "%Y-%m-%d")
    max_date = datetime.strptime(end_train_date, "%Y-%m-%d")
    df_train_val = all_df[all_df['date'].between(min_date, max_date)]

    # Get the train/validation stocks
    train_val_stocks =  np.array(list(set(df_train_val.ticker)))
    train_val_stocks = train_val_stocks[np.random.permutation(len(train_val_stocks))]

    train_stocks = in_train_and_in_test[:int(len(train_val_stocks) * val_train_split)]

    # Split to train and validation
    train_df = df_train_val[df_train_val['ticker'].isin(train_stocks)]
    val_df = df_train_val[~df_train_val['ticker'].isin(train_stocks)]



    ##########################  ma , overfit TODO - remove ###################################################
    if overfit:
        test_df = train_df

    if use_ma:
        def get_ma(df, ks):
            ndf = []
            for n, sdf in df.groupby('ticker'):
                for k in ks:
                    sdf[k] = sdf[k].rolling(window=20).mean().dropna()
                ndf.append(sdf)
            return pd.concat(ndf).dropna()

        train_df = get_ma(train_df, ['close', 'open', 'high', 'low', 'volume'])


        val_df = get_ma(val_df, ['close', 'open', 'high', 'low', 'volume'])

        test_df = get_ma(test_df, ['close', 'open', 'high', 'low', 'volume'])
        test_df_orig = get_ma(test_df_orig, ['close', 'open', 'high', 'low', 'volume'])




    #############################################################################
    pickle.dump(all_norm_factors, open(os.path.join(outputdir, 'norm_factors.pkl'), 'wb'))


    train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))

    val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))

    test_df.to_csv(os.path.join(outputdir, 'test_stocks.csv'))
    test_df_orig.to_csv(os.path.join(outputdir, 'test_df_orig.csv'))



def create_set_from_snp(inputdir: str, outputdir: str,params : Dict,  split_date_factor=0.5 , add_stocks_outof_snp = 0,
                         use_ma = False , overfit = False):
    '''
    Create training set from s&p stocks
     Prepare the 3 sets - train & validation from start_train_date to  end_train_date ,
     test - all stocks start_test_date - end_test_date
    :param inputdir:
    :param outputdir:
    :param split_date_factor: split_date_factor
    :param  add_stocks_outof_snp - stocks to add out of the snp

    :return:
    '''
    stockdir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    os.makedirs(outputdir, exist_ok=True)

    # Get all snp stocks to simulate
    snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/snp500/all_stocks.csv')
    all_dates = np.array(sorted(list(set(snp['date']))))

    start_train_date = all_dates[0]
    end_train_date = all_dates[int(len(all_dates) * split_date_factor)]

    start_test_date = all_dates[int(len(all_dates) * split_date_factor) + 1]
    end_test_date = all_dates[-1]

    # Prepare the 3 sets - train & validation from start_train_date to  end_train_date , test - all stocks start_test_date - end_test_date
    val_train_split = 0.7
    all_stocks_to_train = np.array(list(set(snp['ticker'])))
    all_stocks_to_test = np.array(list(set(snp['ticker'])))
    if add_stocks_outof_snp > 0:
        all_stocks_to_add = get_good_stocks_out_of_snp()
        added = 0
        chosen_stocks_to_add = []
        for stock_name in all_stocks_to_add:
            # Check if there are enough dates to this stock
            stockfile = os.path.join(stockdir, stock_name, 'stockPrice.csv')
            if os.path.exists(stockfile) == False:
                continue
            df = pd.read_csv(stockfile)
            #df = pd.read_excel(os.path.join(stockdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
            time_before = np.sum([pd.to_datetime(start_test_date) > d for d in list(set(pd.to_datetime(df.Date)))  ])
            time_after = np.sum([pd.to_datetime(start_test_date) < d for d in list(set(pd.to_datetime(df.Date)))])
            if ((time_before < params['max_encoder_length'] + 1) | (time_after < params['max_encoder_length'] + 1)):
                continue
            chosen_stocks_to_add.append(stock_name)
            added += 1
            if added > add_stocks_outof_snp:
                break
        all_stocks_to_train = np.hstack([all_stocks_to_train, chosen_stocks_to_add])

    # Create the set - same stocks for test & tain , only different times
    create_set(inputdir, outputdir, params, all_stocks_to_train, all_stocks_to_train, val_train_split, start_train_date, end_train_date,
               start_test_date, end_test_date, use_ma = use_ma , overfit = overfit)

    # Create the set - more stocks in train - may not work well with tft
    # create_set(inputdir, outputdir, params, all_stocks_to_train, all_stocks_to_test, val_train_split, start_train_date, end_train_date,
    #            start_test_date, end_test_date, use_ma = use_ma , overfit = overfit)


if __name__ == "__main__":
    #get_good_stocks_out_of_snp()
    params = get_config()

    np.random.seed(42)
    inputdir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    outputdir = 'C:/Users/dadab/projects/algotrading/data/training/snp_v6'
    create_set_from_snp(inputdir, outputdir, params, split_date_factor=0.5 ,add_stocks_outof_snp = 200 , use_ma = False , overfit = False)

