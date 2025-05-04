import pickle
import pandas as pd
import numpy as np
import os
import shutil
from typing import Dict , List
from sklearn.preprocessing import MinMaxScaler
import pylab as plt

def detect_stocks_with_jumps(inputdir : str,stocks_names: np.array,th = 0.7)->List:
    '''
    Detect stocks with very large jumps in price - omit from training
    '''
    bad_stocks = []
    ngood = 0
    nbad = 0
    for stock_name in stocks_names:

        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        price = np.log(df['close'].values)
        log_price = np.log(np.abs(price)+1)
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
    #plt.show()
    return bad_stocks


def re_arange_df(df: pd.DataFrame,stock_name : str, stock_id :int , norm_factors : Dict):
    # Add some stuff / normalize / so on
    df['time_idx'] = np.arange(len(df))
    df['year'] = [v.astype('datetime64[Y]').astype(int) + 1970 for v in df.Date.values]
    df['month'] = [v.astype('datetime64[M]').astype(int) % 12 + 1 for v in df.Date.values]
    df['day'] = [(v - v.astype('datetime64[M]')).astype(int) + 1 for v in df.Date.values]
    df['stock_name'] = stock_name
    df['stock_id'] = stock_id
    # Rename
    df.rename(
        columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', 'close': 'close', '5. volume': 'volume','Date': 'date'},
        inplace=True)


    df = df[['date', 'year', 'month', 'day', 'open', 'close', 'high', 'low', 'volume', 'stock_name', 'stock_id']]

    # normalize inputs and store the normalization
    # Normalize to by scaling (without offset , for now (?))
    normFact = norm_factors['first_close_scale'] / df['close'].values[0]
    for k in norm_factors['cols_to_pre_normalize_together']:
        df.loc[:,k] =(df[k].astype(float) * normFact).astype(df[k].dtype)
    return df , normFact

def preprocess_data_to_train(inputdir : str, outputdir: str , stock_list_not_to_use : np.array ,  number_of_stocks_to_use : int =None,
                     min_length : int = 150 , val_train_split = 0.7):
    '''
    Prepare data for training - sort out, normalize , rename , add features
    :param inputdir:
    :param outputdir:
    :param stock_list_not_to_use:
    :param min_length: do not take stocks with too little data
    :return:
    '''

    os.makedirs(outputdir, exist_ok=True)
    bad_stocks = np.array(['MTNB','VEON','AVXL','HITI','QXO' , 'ANNAW'])

    good_names = pickle.load(open('C:/Users\dadab\projects/algotrading\data/training/goodstocks.pkl', 'rb'))

    all_stocks = np.array(good_names)
    # bad_stocks = detect_stocks_with_jumps(inputdir, not_to_take)
    # ref1 = pd.read_csv('C:/Users/dadab\projects/algotrading\data/training/dbmed1/train_stocks.csv')
    # names1 = list(set(ref1.stock_name))
    #
    # ref2 = pd.read_csv('C:/Users/dadab\projects/algotrading\data/training/dbmed2/train_stocks.csv')
    #
    # names2 = list(set(ref2.stock_name))
    # all_stocks = np.array(names1 + names2)

    # Get all stocks that can be trained on
    all_stocks = np.array([d for d in os.listdir(inputdir) if
                           (os.path.isdir(os.path.join(all_stock_dir, d)) )])


    # all_stocks = all_stocks[np.random.permutation(len(all_stocks))]
    # bad_stocks = detect_stocks_with_jumps(inputdir, all_stocks,th = 0.3)
    # print('bad_stocks')
    # print(len(bad_stocks))
    # print(bad_stocks)

    # remove bad stocks , add good
   #all_stocks = np.array(list(set(all_stocks) - set(bad_stocks)))
    #
    #
    #all_stocks = np.array(list(set(all_stocks) | set(good_names)))
    print(len(all_stocks))

    # randomize
    all_stocks = all_stocks[np.random.permutation(len(all_stocks))]

    if number_of_stocks_to_use is not None:
        all_stocks = all_stocks[:number_of_stocks_to_use]
    print('all_stocks  ',len(all_stocks))
    #pickle.dump(all_stocks, open('C:/Users\dadab\projects/algotrading\data/training/goodstocks_3.pkl', 'wb'))

    norm_factors = {}
    norm_factors['cols_to_pre_normalize_together'] =  ['open', 'high', 'low', 'close', 'volume']
    norm_factors['first_close_scale'] = 1.0 # the scale is determined by the first close value

    # Split df to train and validation
    train_stocks = all_stocks[:int(len(all_stocks)*val_train_split)]
    val_stocks = all_stocks[int(len(all_stocks)*val_train_split):]

    dfs = []
    for i, stock_name in enumerate(train_stocks):
        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        if (len(df) < min_length):
            continue
        df, normFact = re_arange_df(df, stock_name,i, norm_factors)
        dfs.append(df)
        norm_factors[stock_name + 'normFact'] = normFact

    train_df = pd.concat(dfs, ignore_index=True)
    # Fill NA values
    train_df = train_df.fillna(0)

    # Create time index - ensure each ticker starts from 0
    train_df =train_df.sort_values(['stock_name', 'date'])
    train_df['time_idx'] = train_df.groupby('stock_name').cumcount()

    # Drop rows with missing values
    train_df= train_df.dropna()

    # Rescale volume and other features on all data
    cols_to_normalize = ['volume']
    for k in cols_to_normalize:
        norm_factors[k + 'offset'] =  np.min(train_df['volume'].values)
        norm_factors[k + 'scale'] = 1.0 / (np.max(train_df['volume'].values) - norm_factors[k + 'offset'])
        train_df[k] = (train_df[k] -    norm_factors[k + 'offset']) * norm_factors[k + 'scale']

    # Save training
    train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))

    # Get validation set
    dfs = []
    for i, stock_name in enumerate(val_stocks):
        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        if (len(df) < min_length):
            continue
        df, normFact  = re_arange_df(df, stock_name,i, norm_factors)
        norm_factors[stock_name + 'normFact'] = normFact
        dfs.append(df)

    val_df = pd.concat(dfs, ignore_index=True)
    # Fill NA values
    val_df = train_df.fillna(0)

    # Create time index - ensure each ticker starts from 0
    val_df =val_df.sort_values(['stock_name', 'date'])
    val_df['time_idx'] = val_df.groupby('stock_name').cumcount()

    #Scale with t
    for k in cols_to_normalize:
        val_df[k] = (val_df[k] -    norm_factors[k + 'offset']) * norm_factors[k + 'scale']

    # Save validation
    val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))

    # Save normalization
    pickle.dump(norm_factors, open(os.path.join(outputdir, 'norm_factors.pkl'), 'wb'))
if __name__ == "__main__":
    np.random.seed(42)


    all_stock_dir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    datadir ='C:/Users/dadab/projects/algotrading/data/training/dbsmall'
    #
    snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/tickers/sp500_stocks.csv')

    preprocess_data_to_train(all_stock_dir, datadir,sorted(snp['Ticker'].values),number_of_stocks_to_use=5)

    # create_training_set(all_stock_dir, datadir,sorted(snp['Ticker'].values) , 40)

    #
    # create_index("C:/Users\dadab\projects/algotrading\data/tickers", datadir,sorted(snp['Ticker'].values) ,
    #              filter_by_length = False ,  )
