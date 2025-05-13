import pickle
import pandas as pd
import numpy as np
import os
import shutil
from typing import Dict, List
from datetime import datetime
from copy import deepcopy
import pylab as plt


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




    # normalize inputs and store the normalization

    # Normalize to by scaling (without offset , for now (?))
    normFact = norm_factors['first_close_scale'] / df['close'].values[0]
    for k in norm_factors['cols_to_pre_normalize_together']:
        df.loc[:, k] = (df[k].astype(float) * normFact).astype(df[k].dtype)

    # TODO - add more features


    return df, normFact


def get_normalized_training_data(inputdir, stock_list_to_use, first_ind = 0, minDate=None, maxDate=None,
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
    norm_factors['cols_to_pre_normalize_together'] = ['open', 'high', 'low', 'close', 'volume']
    norm_factors['first_close_scale'] = 1.0  # the scale is determined by the first close value

    dfs = []
    dfs_raw = []
    ind = first_ind
    for ticker in stock_list_to_use:
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

    # Rescale volume and other features on all data
    cols_to_normalize = ['volume']
    for k in cols_to_normalize:
        norm_factors[k + 'offset'] = np.min(train_df['volume'].values)
        norm_factors[k + 'scale'] = 1.0 / (np.max(train_df['volume'].values) - norm_factors[k + 'offset'])
        train_df[k] = (train_df[k] - norm_factors[k + 'offset']) * norm_factors[k + 'scale']

    return train_df, norm_factors, raw_df


# def preprocess_data_to_train(inputdir : str, outputdir: str ,stock_list_not_to_use : []  ,  stock_list_to_use : np.array  = None,  number_of_stocks_to_use : int =None,
#                      min_length : int = 150 , val_train_split = 0.7 ,  split_date_factor = 0.5):
#     '''
#     Prepare data for training - sort out, normalize , rename , add features
#     :param inputdir: directory of the raw ticker data
#     :param outputdir:
#     :param stock_list_not_to_use:
#     :param min_length: do not take stocks with too little data
#     :return:
#     '''
#
#     os.makedirs(outputdir, exist_ok=True)
#
#     if len(stock_list_to_use) == 0:
#         # Take a given set of
#
#         # Get all stocks that can be trained on
#         all_stocks = np.array([d for d in os.listdir(inputdir) if
#                                (os.path.isdir(os.path.join(inputdir, d)) )])
#
#
#         all_stocks = all_stocks[np.random.permutation(len(all_stocks))]
#
#         # Get the nad stocks
#         bad_stocks = detect_stocks_with_jumps(inputdir, all_stocks,th = 0.3)
#         stock_list_not_to_use = bad_stocks + stock_list_not_to_use
#         #bad_stocks = np.array(['MTNB', 'VEON', 'AVXL', 'HITI', 'QXO', 'ANNAW'])
#
#         # print('bad_stocks')
#         # print(len(bad_stocks))
#         # print(bad_stocks)
#
#         # remove bad stocks , add good
#
#         #all_stocks = np.array(list(set(all_stocks) | set(good_names)))
#     else:
#         all_stocks = stock_list_to_use
#
#     all_stocks = np.array(list(set(all_stocks) - set(stock_list_not_to_use)))
#
#     print(len(all_stocks))
#
#     # randomize
#     all_stocks = all_stocks[np.random.permutation(len(all_stocks))]
#
#     if number_of_stocks_to_use is not None:
#         all_stocks = all_stocks[:number_of_stocks_to_use]
#     print('all_stocks  ',len(all_stocks))
#
#
#
#     norm_factors = {}
#     norm_factors['cols_to_pre_normalize_together'] =  ['open', 'high', 'low', 'close', 'volume']
#     norm_factors['first_close_scale'] = 1.0 # the scale is determined by the first close value
#
#     # Split df to train and validation
#     train_stocks = all_stocks[:int(len(all_stocks)*val_train_split)]
#     val_stocks = all_stocks[int(len(all_stocks)*val_train_split):]
#
#     train_df ,train_norm_factors ,_ =  get_normalized_training_data(inputdir , train_stocks )
#     #save data
#     train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))
#     # Save normalization
#     pickle.dump(train_norm_factors, open(os.path.join(outputdir, 'train_norm_factors.pkl'), 'wb'))
#
#
#     val_df, val_norm_factors,_ = get_normalized_training_data(inputdir, val_stocks)
#     #save data
#     val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))
#     # Save normalization
#     pickle.dump(val_norm_factors, open(os.path.join(outputdir, 'val_norm_factors.pkl'), 'wb'))
#
#     # dfs = []
#     # for i, stock_name in enumerate(train_stocks):
#     #     df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
#     #     if (len(df) < min_length):
#     #         continue
#     #     df, normFact = re_arange_df(df, stock_name,i, norm_factors)
#     #     dfs.append(df)
#     #     norm_factors[stock_name + 'normFact'] = normFact
#     #
#     # train_df = pd.concat(dfs, ignore_index=True)
#     # # Fill NA values
#     # train_df = train_df.fillna(0)
#     #
#     # # Create time index - ensure each ticker starts from 0
#     # train_df =train_df.sort_values(['stock_name', 'date'])
#     # train_df['time_idx'] = train_df.groupby('stock_name').cumcount()
#     #
#     # # Drop rows with missing values
#     # train_df= train_df.dropna()
#     #
#     # # Rescale volume and other features on all data
#     # cols_to_normalize = ['volume']
#     # for k in cols_to_normalize:
#     #     norm_factors[k + 'offset'] =  np.min(train_df['volume'].values)
#     #     norm_factors[k + 'scale'] = 1.0 / (np.max(train_df['volume'].values) - norm_factors[k + 'offset'])
#     #     train_df[k] = (train_df[k] -    norm_factors[k + 'offset']) * norm_factors[k + 'scale']
#     #
#     # # Save training
#     # train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))
#     #
#     # # Get validation set
#     # dfs = []
#     # for i, stock_name in enumerate(val_stocks):
#     #     df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
#     #     if (len(df) < min_length):
#     #         continue
#     #     df, normFact  = re_arange_df(df, stock_name,i, norm_factors)
#     #     norm_factors[stock_name + 'normFact'] = normFact
#     #     dfs.append(df)
#     #
#     # val_df = pd.concat(dfs, ignore_index=True)
#     # # Fill NA values
#     # val_df = val_df.fillna(0)
#     #
#     # # Create time index - ensure each ticker starts from 0
#     # val_df =val_df.sort_values(['stock_name', 'date'])
#     # val_df['time_idx'] = val_df.groupby('stock_name').cumcount()
#     #
#     # #Scale with t
#     # for k in cols_to_normalize:
#     #     val_df[k] = (val_df[k] -    norm_factors[k + 'offset']) * norm_factors[k + 'scale']
#     #
#     # # Save validation
#     # val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))
#     #
#     # # Save normalization
#     # pickle.dump(norm_factors, open(os.path.join(outputdir, 'norm_factors.pkl'), 'wb'))

def get_good_stocks_out_of_snp(   recalc = False):
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


# def create_set_from_stocks_out_and_in_snp(inputdir: str, outputdir: str, split_date_factor=0.5):
#     '''
#      Create set that include sstocks out of the snp in the tranining and validation , and snp only in the test
#      :param inputdir:
#      :param outputdir:
#      :param split_date_factor:
#      :return:
#     '''
#
#     os.makedirs(outputdir, exist_ok=True)
#
#     # Get "knowen " good stocks
#     good_names = pickle.load(open('C:/Users/dadab/projects/algotrading/data/training/goodstocks_v0.pkl', 'rb'))
#     ref2 = pd.read_csv('C:/Users/dadab/projects/algotrading/data/training/obsolete/dbmed4/train_stocks.csv')
#     names2 = list(set(ref2.stock_name))
#     all_stocks = np.array(good_names + names2)
#
#
#     # Get all snp stocks to simulate
#     snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/snp500/all_stocks.csv')
#     all_dates = np.array(sorted(list(set(snp['date']))))
#
#     start_train_date = all_dates[0]
#     end_train_date = all_dates[int(len(all_dates) * split_date_factor)]
#
#     start_test_date = all_dates[int(len(all_dates) * split_date_factor) + 1]
#     end_test_date = all_dates[-1]
#     snp_stocks = np.array(list(set(snp['ticker'])))
#     # add all stocks
#     all_stocks = np.array(list(set(all_stocks) | set(snp_stocks)))
#
#     # Prepare the 3 sets - train & validation from start_train_date to  end_train_date , test - all stocks start_test_date - end_test_date
#     val_train_split = 0.7
#
#     create_set(inputdir, outputdir, all_stocks, snp_stocks, val_train_split, start_train_date, end_train_date,
#                start_test_date,
#                end_test_date)
#

def create_set(inputdir: str, outputdir: str, all_training_stocks: np.array, all_test_stocks: np.array, val_train_split,
               start_train_date, end_train_date, start_test_date, end_test_date , use_ma = False , overfit = False):
    '''
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

    in_train_and_in_test = all_test_stocks[np.random.permutation(len(all_test_stocks))]
    in_train_but_not_in_test = in_train_but_not_in_test[np.random.permutation(len(in_train_but_not_in_test))]

    if overfit:
        # overfit
        in_train_and_in_test = in_train_and_in_test[:3]
        in_train_but_not_in_test= in_train_but_not_in_test[:3]

    train_stocks = in_train_and_in_test[:int(len(in_train_and_in_test) * val_train_split)]
    val_stocks = in_train_and_in_test[int(len(in_train_and_in_test) * val_train_split):]

    test_stocks = in_train_and_in_test


    # Get all data
    all_df, all_norm_factors, all_df_orig = get_normalized_training_data(inputdir, in_train_and_in_test,0)
    min_date = datetime.strptime(start_train_date, "%Y-%m-%d")
    max_date = datetime.strptime(end_train_date, "%Y-%m-%d")
    df_train_val = all_df[all_df['date'].between(min_date, max_date)]
    # Split train and validation
    train_df = df_train_val[df_train_val['ticker'].isin(train_stocks)]  # Only rows where category is A or B
    val_df = df_train_val[~df_train_val['ticker'].isin(train_stocks)]

    # Get  test set on later times
    min_date = datetime.strptime(start_test_date, "%Y-%m-%d")
    max_date = datetime.strptime(end_test_date, "%Y-%m-%d")
    test_df = all_df[all_df['date'].between(min_date, max_date)]
    test_df_orig = all_df_orig[all_df_orig['date'].between(min_date, max_date)]







    #split to train, validation & test




    # # Get the train + val data on training dates
    # train_df, train_norm_factors, _ = get_normalized_training_data(inputdir, train_stocks,0, start_train_date,
    #                                                                end_train_date)
    #
    # val_df, val_norm_factors, _ = get_normalized_training_data(inputdir, val_stocks,np.max(list(set(train_df.stock_id))) + 1,  start_train_date, end_train_date)


    # Add extra stocks only to the train and validation sets
    if(len(in_train_but_not_in_test) > 10):
        # Split extra data
        extra_df, extra_norm_factors, _ = get_normalized_training_data(inputdir, in_train_and_in_test, 0,
                                                                           start_train_date,
                                                                           end_train_date)

        extra_train_stocks = in_train_but_not_in_test[:int(len(in_train_but_not_in_test) * val_train_split)]
        extra_val_stocks = in_train_but_not_in_test[int(len(in_train_but_not_in_test) * val_train_split):]

        # Add extra data
        extra_train_df, extra_train_norm_factors, _ = get_normalized_training_data(inputdir, extra_train_stocks,  np.max(list(set(val_df.stock_id))) + 1, start_train_date,
                                                                       end_train_date)

        extra_val_df, extra_val_norm_factors, _ = get_normalized_training_data(inputdir, extra_val_stocks,
                                                                   np.max(list(set(extra_train_df.stock_id))) + 1,
                                                                   start_train_date, end_train_date)

        # merge
        # train_norm_factors.update(extra_train_norm_factors)
        # val_norm_factors.update(extra_val_norm_factors)
        #
        # train_df = pd.concat([train_df, extra_train_df]).reset_index(drop=True)
        # val_df = pd.concat([val_df, extra_val_df]).reset_index(drop=True)

    # Translation of id to name in training set , so the test stocks will be compatible
    # ticker_to_id = {}
    # for ticker, df in train_df.groupby('ticker'):
    #     stock_ids = set(df.stock_id.values)
    #     assert (len(stock_ids) == 1)
    #     ticker_to_id[ticker] = list(stock_ids)[0]
    # for ticker, df in val_df.groupby('ticker'):
    #     stock_ids = set(df.stock_id.values)
    #     assert (len(stock_ids) == 1)
    #     ticker_to_id[ticker] = list(stock_ids)[0]
    #
    #
    # test_df, test_norm_factors, test_df_orig = get_normalized_training_data(inputdir, test_stocks,0,  start_test_date,
    #                                                                         end_test_date , ticker_to_id=ticker_to_id)


    ########################## add ma , overfit TODO - remove ###################################################
    if overfit:
        # Get  test set on later times
        min_date = datetime.strptime(start_train_date, "%Y-%m-%d")
        max_date = datetime.strptime(end_train_date, "%Y-%m-%d")
        test_df = all_df[all_df['date'].between(min_date, max_date)]
        test_df_orig = all_df_orig[all_df['date'].between(min_date, max_date)]



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

    train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))
    pickle.dump(all_norm_factors, open(os.path.join(outputdir, 'train_stocks_norm_factors.pkl'), 'wb'))

    val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))
    pickle.dump(all_norm_factors, open(os.path.join(outputdir, 'val_stocks_norm_factors.pkl'), 'wb'))

    test_df.to_csv(os.path.join(outputdir, 'test_stocks.csv'))
    test_df_orig.to_csv(os.path.join(outputdir, 'test_df_orig.csv'))
    pickle.dump(all_norm_factors, open(os.path.join(outputdir, 'test_stocks_norm_factors.pkl'), 'wb'))


def create_set_from_snp(inputdir: str, outputdir: str, split_date_factor=0.5 , add_stocks_outof_snp = 0,
                         use_ma = False , overfit = False):
    '''
    Create training set from s&p
     Prepare the 3 sets - train & validation from start_train_date to  end_train_date ,
     test - all stocks start_test_date - end_test_date
    :param inputdir:
    :param outputdir:
    :param split_date_factor: split_date_factor
    :param  add_stocks_outof_snp - stocks to add out of the snp

    :return:
    '''

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
        stocks_to_add = get_good_stocks_out_of_snp()
        stocks_to_add = stocks_to_add[:add_stocks_outof_snp]
        all_stocks_to_train = np.hstack([all_stocks_to_train, stocks_to_add])

    create_set(inputdir, outputdir, all_stocks_to_train, all_stocks_to_test, val_train_split, start_train_date, end_train_date,
               start_test_date, end_test_date, use_ma = use_ma , overfit = overfit)


if __name__ == "__main__":
    #get_good_stocks_out_of_snp()
    np.random.seed(42)
    inputdir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    outputdir = 'C:/Users/dadab/projects/algotrading/data/training/snp_v1_ma20'
    create_set_from_snp(inputdir, outputdir, split_date_factor=0.5 ,add_stocks_outof_snp = 0 , use_ma = True , overfit = False)

    # create_set_from_stocks_out_and_in_snp(inputdir, outputdir  ,split_date_factor= 0.5 )

    # outputdir = 'C:/Users/dadab/projects/algotrading/data/training/snp_overfit'
    #
    # create_set_from_snp(inputdir, outputdir ,split_date_factor= 0.5 )

