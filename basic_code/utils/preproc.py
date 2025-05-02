import pickle
import pandas as pd
import numpy as np
import os
import shutil
from typing import Dict
from sklearn.preprocessing import MinMaxScaler

def prerprocess_data(datadir : str , minLengthtoUse :int = 300):
    '''
    Prepare data for work -
    - Filter out stocks that does not have enough information
    - Take only dates that has data from all stocks
    - Create reference index - average of all stocks
    - Save the reference index and the common stocks data frame
    :param datadir: input directory
    :param minLengthtoUse:  Minimal number of dates in a stock file directory
    '''


    dirnames = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
    dfs = []
    for dirname in dirnames:
        filename = os.path.join(datadir, dirname, 'stockPrice.xlsx')
        df = pd.read_excel(filename, engine='openpyxl')
        print(f'{dirname} from {np.min(df.Date)}  to  {np.max(df.Date)}  {len(df)}')
        if(len(df)  < minLengthtoUse):
            continue
        # Add some features
        df['name'] = dirname

        dfs.append(df)
    df_all = pd.concat(dfs)

    # Take dates that has all stocks information
    Nstocks = len(set(df_all.name))
    filtered_df = []

    for date, df in df_all.groupby('Date'):
        if(len(df) ==Nstocks):
            filtered_df.append(df)



    df_all = pd.concat(filtered_df).reset_index()

    if (len(set(df_all.Date)) < minLengthtoUse):
        print('Error : no enough dates')
        return

    print(f' preroc {datadir} #stocks {len(set(df_all.name))} #dates {len(set(df_all.Date))}  from {np.min(df_all.Date)} to {np.max(df_all.Date)} ')
    # Save data
    df_all.to_csv(os.path.join(datadir, 'all_stocks.csv'))
    # Get & Save  average index
    avgdata = get_average_stock(df_all)
    avgdata.to_csv(os.path.join(datadir, 'reference_index.csv'))



def create_index(inputdir : str, outputdir: str , stock_list : np.array, filter_by_length : bool = False , referenceDates = None):
    '''
    Copy part of the stocks
    :param inputdir: 
    :param outputdir: 
    :param stock_list_path:
    :param use_stocks_in_list - if true , force all stocks to have at least referenceDates
    :param filter_by_length - if true , force all stocks to have at least referenceDates
    :param referenceDates - reference dates that all stocks must have , if None - take it from the first stock

    :return: 
    '''
    os.makedirs(outputdir,exist_ok=True)

    if(filter_by_length):
        nfiles  = 0
        for sname in stock_list:
            if os.path.isdir(os.path.join(inputdir, sname)):
                df = pd.read_excel(os.path.join(inputdir, sname, 'stockPrice.xlsx'), engine='openpyxl')
                if referenceDates is None:
                    referenceDates = df.Date.values
                    shutil.copytree(os.path.join(inputdir, sname), os.path.join(outputdir, sname))
                    print(f'{nfiles} copy {sname} {len( df.Date.values)}')
                    nfiles += 1

                else:
                    if len(set(referenceDates)- set(df.Date.values)) == 0:
                        print(f'{nfiles} copy {sname}  {len( df.Date.values)}')
                        nfiles += 1
                        shutil.copytree(os.path.join(inputdir, sname), os.path.join(outputdir, sname))
                    else:
                        print(f'{sname} does not fit  {len( df.Date.values)}')
    else:
        for sname in stock_list:
            if os.path.isdir(os.path.join(inputdir, sname)):
                shutil.copytree(os.path.join(inputdir, sname),os.path.join(outputdir, sname) )
            else:
                print(f"{ os.path.join(inputdir, sname)} not found ")

def get_average_stock(dfi : pd.DataFrame)->pd.DataFrame:
    '''
    Average all stocks with equal weights
    Normalization - for each stock, set the first closing price will be 100
    :return: average dataframe
    '''
    reference_key = [k for k in dfi.keys() if 'close' in  k ][0]
    keys_to_average =  ['1. open', '2. high', '3. low',reference_key , '5. volume']

    # Normalize
    df = dfi.copy()
    refData = np.min(df.Date)
    stock_names = set(df.name)
    for stock_name in stock_names:
       # normalize so first closing price will be 100
       normFact = 100 / df[(df.name == stock_name) & (df.Date == refData)][reference_key].values[0]
       for k in keys_to_average:
            df.loc[df.name == stock_name, k] = df[df.name == stock_name][k].astype(float) * normFact

    # average on all stocks per time
    res = []
    for date, df_date in df.groupby('Date'):
        r = {'Date': date, 'name': 'average'}
        for k in keys_to_average:
            r[k] = df_date[k].mean()
        res.append(r)
    return pd.DataFrame(res)

def re_arange_df(df: pd.DataFrame,stock_name : str,  norm_factors : Dict):
    # Add some stuff ...
    df['time_idx'] = np.arange(len(df))
    df['year'] = [v.astype('datetime64[Y]').astype(int) + 1970 for v in df.Date.values]
    df['month'] = [v.astype('datetime64[M]').astype(int) % 12 + 1 for v in df.Date.values]
    df['day'] = [(v - v.astype('datetime64[M]')).astype(int) + 1 for v in df.Date.values]
    df['stock_name'] = stock_name
    # Rename
    df['open'] = df['1. open']
    df['high'] = df['2. high']
    df['low'] = df['3. low']
    df['volume'] = df['5. volume']

    df = df[['time_idx', 'year', 'month', 'day', 'open', 'close', 'high', 'low', 'volume', 'stock_name']]

    # scaler = MinMaxScaler()
    # values = scaler.fit_transform(df[cols_to_pre_normalize_together])

    df = df[['time_idx', 'year', 'month', 'day', 'stock_name', 'open', 'close', 'high', 'low', 'volume']]
    # normalize inputs and store the normalization
    # Normalize to by scaling (without offset , for now (?))
    normFact = norm_factors['first_close_scale'] / df['close'].values[0]
    for k in norm_factors['cols_to_pre_normalize_together']:
        df[k] = df[k].astype(float) * normFact
    return df , normFact

def preprocess_data_to_train(inputdir : str, outputdir: str , stock_list_not_to_use : np.array ,  number_of_stocks_to_use : int =None,
                     min_length : int = 150):
    '''
    Prepare data for training - sort out, normalize , rename , add features
    :param inputdir:
    :param outputdir:
    :param stock_list_not_to_use:
    :param min_length: dont take stocks with too little data
    :return:
    '''
    os.makedirs(outputdir, exist_ok=True)
    # get all stocks that can be trained on
    all_stocks = np.array([d for d in os.listdir(inputdir) if
                           (os.path.isdir(os.path.join(all_stock_dir, d)) & (d not in stock_list_not_to_use))])
    all_stocks = all_stocks[np.random.permutation(len(all_stocks))]

    if number_of_stocks_to_use is not None:
        all_stocks = all_stocks[:number_of_stocks_to_use]


    norm_factors = {}
    norm_factors['cols_to_pre_normalize_together'] =  ['open', 'high', 'low', 'close', 'volume']
    norm_factors['first_close_scale'] = 1.0 # the scale is determined by the first close value

    # Split df to train and validation
    train_stocks = all_stocks[:int(len(all_stocks)*0.7)]
    val_stocks = all_stocks[int(len(all_stocks)*0.7):]

    dfs = []
    for stock_name in train_stocks:
        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        if (len(df) < min_length):
            continue
        df, normFact = re_arange_df(df, stock_name, norm_factors)
        dfs.append(df)
        norm_factors[stock_name + 'normFact'] = normFact

    train_df = pd.concat(dfs)


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
    for stock_name in val_stocks:
        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        if (len(df) < min_length):
            continue
        df, normFact  = re_arange_df(df, stock_name, norm_factors)
        norm_factors[stock_name + 'normFact'] = normFact
        dfs.append(df)

    val_df = pd.concat(dfs)
    #Scale with t
    for k in cols_to_normalize:
        val_df[k] = (val_df[k] -    norm_factors[k + 'offset']) * norm_factors[k + 'scale']

    # Save validation
    val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))

    # Save normalization
    pickle.dump(norm_factors, open(os.path.join(outputdir, 'norm_factors.pkl'), 'wb'))

if __name__ == "__main__":
    # datadir ='C:/Users\dadab\projects/algotrading\data/snp500'
    #
    # snp = pd.read_csv('C:/Users\dadab\projects/algotrading\data/tickers\sp500_stocks.csv')
    #
    # create_index("C:/Users\dadab\projects/algotrading\data/tickers", datadir,sorted(snp['Ticker'].values) ,
    #              filter_by_length = False ,  )
    # prerprocess_data(datadir)



    all_stock_dir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    datadir ='C:/Users/dadab/projects/algotrading/data/training/dbbig'
    #
    snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/tickers/sp500_stocks.csv')

    preprocess_data_to_train(all_stock_dir, datadir,sorted(snp['Ticker'].values),number_of_stocks_to_use=1000)

    # create_training_set(all_stock_dir, datadir,sorted(snp['Ticker'].values) , 40)

    #
    # create_index("C:/Users\dadab\projects/algotrading\data/tickers", datadir,sorted(snp['Ticker'].values) ,
    #              filter_by_length = False ,  )

