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
            ngood += 1
            print(f' {stock_name} is good {ngood} {nbad}')

        # plt.figure()
        # plt.plot(price)
        # plt.plot(log_price)
        # plt.plot(np.diff(log_price))
        # plt.plot(np.abs(np.diff(np.diff(np.diff(log_price)))))
        # plt.title(stock_name)
    print(bad_stocks)
    # plt.show()
    return bad_stocks


def re_arange_df(df: pd.DataFrame, ticker: str, stock_id: int, norm_factors: Dict):
    # Add some stuff / normalize / so on

    df['time_idx'] = np.arange(len(df))
    df['year'] = [v.astype('datetime64[Y]').astype(int) + 1970 for v in df.date.values]
    df['month'] = [v.astype('datetime64[M]').astype(int) % 12 + 1 for v in df.date.values]
    df['day'] = [(v - v.astype('datetime64[M]')).astype(int) + 1 for v in df.date.values]
    df['ticker'] = ticker
    df['stock_id'] = stock_id

    df = df[['date', 'year', 'month', 'day', 'open', 'close', 'high', 'low', 'volume', 'ticker', 'stock_id']]
    # TODO - add more inputs
    # normalize inputs and store the normalization

    # Normalize to by scaling (without offset , for now (?))
    normFact = norm_factors['first_close_scale'] / df['close'].values[0]
    for k in norm_factors['cols_to_pre_normalize_together']:
        df.loc[:, k] = (df[k].astype(float) * normFact).astype(df[k].dtype)

    return df, normFact


def get_normalized_training_data(inputdir, stock_list_to_use, stock_ids,  minDate=None, maxDate=None,
                                 min_length: int = 150,
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
    for ticker, i  in zip(stock_list_to_use , stock_ids):
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
            datetime.strptime(minDate, "%Y-%m-%d")

            df = df[df['date'].between(min_date, max_date)]

        if (len(df) < min_length):
            continue
        # Store non-normalized data
        df_raw = deepcopy(df)
        df_raw['ticker'] = ticker
        dfs_raw.append(df_raw)

        df, normFact = re_arange_df(df, ticker, i, norm_factors)
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

def create_set_from_stocks_out_and_in_snp(inputdir: str, outputdir: str, split_date_factor=0.5):
    '''
     Create set that include sstocks out of the snp in the tranining and validation , and snp only in the test
     :param inputdir:
     :param outputdir:
     :param split_date_factor:
     :return:
    '''

    os.makedirs(outputdir, exist_ok=True)

    # Get "knowen " good stocks
    good_names = pickle.load(open('C:/Users/dadab/projects/algotrading/data/training/goodstocks.pkl', 'rb'))
    ref2 = pd.read_csv('C:/Users/dadab/projects/algotrading/data/training/obsolete/dbmed4/train_stocks.csv')
    names2 = list(set(ref2.stock_name))
    all_stocks = np.array(good_names + names2)

    # Get all snp stocks to simulate
    snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/snp500/all_stocks.csv')
    all_dates = np.array(sorted(list(set(snp['date']))))

    start_train_date = all_dates[0]
    end_train_date = all_dates[int(len(all_dates) * split_date_factor)]

    start_test_date = all_dates[int(len(all_dates) * split_date_factor) + 1]
    end_test_date = all_dates[-1]
    snp_stocks = np.array(list(set(snp['ticker'])))
    # add all stocks
    all_stocks = np.array(list(set(all_stocks) | set(snp_stocks)))

    # Prepare the 3 sets - train & validation from start_train_date to  end_train_date , test - all stocks start_test_date - end_test_date
    val_train_split = 0.7

    create_set(inputdir, outputdir, all_stocks, snp_stocks, val_train_split, start_train_date, end_train_date,
               start_test_date,
               end_test_date)


def create_set(inputdir: str, outputdir: str, all_stocks: np.array, all_test_stocks: np.array, val_train_split,
               start_train_date, end_train_date, start_test_date, end_test_date):
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

    # randomize
    all_stocks = all_stocks[np.random.permutation(len(all_stocks))]

    train_stocks = all_stocks[:int(len(all_stocks) * val_train_split)]
    val_stocks = all_stocks[int(len(all_stocks) * val_train_split):]
    train_stocks_ids = np.arange(len(train_stocks))
    val_stocks_ids = np.arange(len(val_stocks))+train_stocks_ids[-1]


    # Get the train + val data on training dates
    train_df, train_norm_factors, _ = get_normalized_training_data(inputdir, train_stocks,train_stocks_ids, start_train_date,
                                                                   end_train_date)

    val_df, val_norm_factors, _ = get_normalized_training_data(inputdir, val_stocks,val_stocks_ids, start_train_date, end_train_date)
    # test_df, test_norm_factors, test_df_orig = get_normalized_training_data(inputdir, all_test_stocks, start_test_date,
    #                                                                         end_test_date)
    test_df, test_norm_factors, test_df_orig = get_normalized_training_data(inputdir, train_stocks,val_stocks_ids, start_test_date,
                                                                            end_test_date)

    ########################## add ma TODO - remove ###################################################
    def get_ma(df, ks):
        ndf = []
        for n, sdf in df.groupby('ticker'):
            for k in ks:
                sdf[k] = sdf[k].rolling(window=20).mean().dropna()
            ndf.append(sdf)
        return pd.concat(ndf).dropna()

    train_df = get_ma(train_df, ['close', 'open', 'high', 'low', 'volume'])
    train_df = train_df[(train_df.stock_id == 0) |(train_df.stock_id == 1)  ]
    val_df = train_df
    test_df = get_ma(test_df, ['close', 'open', 'high', 'low', 'volume'])
    test_df_orig = get_ma(test_df_orig, ['close', 'open', 'high', 'low', 'volume'])

    #############################################################################

    train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))
    pickle.dump(train_norm_factors, open(os.path.join(outputdir, 'train_stocks_norm_factors.pkl'), 'wb'))

    val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))
    pickle.dump(val_norm_factors, open(os.path.join(outputdir, 'val_stocks_norm_factors.pkl'), 'wb'))

    test_df.to_csv(os.path.join(outputdir, 'test_stocks.csv'))
    test_df_orig.to_csv(os.path.join(outputdir, 'test_df_orig.csv'))
    pickle.dump(test_norm_factors, open(os.path.join(outputdir, 'test_stocks_norm_factors.pkl'), 'wb'))


def create_set_from_snp(inputdir: str, outputdir: str, split_date_factor=0.5):
    '''
    Create training set from s&p
     Prepare the 3 sets - train & validation from start_train_date to  end_train_date ,
     test - all stocks start_test_date - end_test_date
    :param inputdir:
    :param outputdir:
    :param split_date_factor: split_date_factor

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
    all_stocks = np.array(list(set(snp['ticker'])))
    create_set(inputdir, outputdir, all_stocks, all_stocks, val_train_split, start_train_date, end_train_date,
               start_test_date, end_test_date)

    # # randomize
    # all_stocks = all_stocks[np.random.permutation(len(all_stocks))]
    #
    #
    # train_stocks = all_stocks[:int(len(all_stocks)*val_train_split)]
    # val_stocks = all_stocks[int(len(all_stocks)*val_train_split):]
    #
    #
    # #Get the train + val data on training dates
    # train_df, train_norm_factors,_  = get_normalized_training_data(inputdir, train_stocks , start_train_date , end_train_date )
    # val_df, val_norm_factors ,_ = get_normalized_training_data(inputdir, val_stocks, start_train_date, end_train_date)
    # test_df, test_norm_factors ,test_df_orig = get_normalized_training_data(inputdir, train_stocks , start_test_date , end_test_date )
    #
    # ########################## TODO - remove ###################################################
    # def get_ma(df,ks):
    #     ndf = []
    #     for n, sdf in df.groupby('ticker'):
    #         for k in ks:
    #             sdf[k] = sdf[k].rolling(window=20).mean().dropna()
    #         ndf.append(sdf)
    #     return pd.concat(ndf).dropna()
    #
    # train_df = get_ma(train_df, ['close', 'open', 'high', 'low', 'volume'])
    # train_df = train_df[train_df.ticker == train_df.ticker.values[0]]
    # val_df = train_df
    # test_df = get_ma(test_df, ['close', 'open', 'high', 'low', 'volume'])
    # test_df_orig = get_ma(test_df_orig, ['close', 'open', 'high', 'low', 'volume'])
    #
    #
    # #############################################################################
    #
    #
    #
    #
    # train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))
    # pickle.dump(train_norm_factors, open(os.path.join(outputdir, 'train_stocks_norm_factors.pkl'), 'wb'))
    #
    # val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))
    # pickle.dump(val_norm_factors, open(os.path.join(outputdir, 'val_stocks_norm_factors.pkl'), 'wb'))
    #
    # test_df.to_csv(os.path.join(outputdir, 'test_stocks.csv'))
    # test_df_orig.to_csv(os.path.join(outputdir, 'test_df_orig.csv'))
    # pickle.dump(test_norm_factors, open(os.path.join(outputdir, 'test_stocks_norm_factors.pkl'), 'wb'))


if __name__ == "__main__":
    np.random.seed(42)
    inputdir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    outputdir = 'C:/Users/dadab/projects/algotrading/data/training/snp_plus_ma20_5'
    create_set_from_snp(inputdir, outputdir, split_date_factor=0.5)
    # create_set_from_stocks_out_and_in_snp(inputdir, outputdir  ,split_date_factor= 0.5 )

    # outputdir = 'C:/Users/dadab/projects/algotrading/data/training/snp_overfit'
    #
    # create_set_from_snp(inputdir, outputdir ,split_date_factor= 0.5 )

