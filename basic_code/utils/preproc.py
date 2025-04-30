import pylab as plt
import pandas as pd
import numpy as np
import os
import shutil

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

def  dataset_preproc_conversion(inputdir : str,stocks_to_use : np.array, min_length : int ):
    dts = []
    for stock_name in stocks_to_use:
        df = pd.read_excel(os.path.join(inputdir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        if (len(df) < min_length):
            continue
        # Add some stuff ...
        df['time_idx'] = np.arange(len(df))  # To do - make this a global index
        df['year'] = [v.astype('datetime64[Y]').astype(int) + 1970 for v in df.Date.values]
        df['month'] = [v.astype('datetime64[M]').astype(int) % 12 + 1 for v in df.Date.values]
        df['day'] = [(v - v.astype('datetime64[M]')).astype(int) + 1 for v in df.Date.values]
        df['log_close'] = np.log(df['close'] + 1)
        df['stock_name'] = stock_name
        df['open'] = df['1. open']
        df['high'] = df['2. high']
        df['low'] = df['3. low']
        df['volume'] = df['5. volume']
        df = df[
            ['time_idx', 'year', 'month', 'day', 'open', 'close', 'high', 'low', 'volume', 'log_close', 'stock_name']]
        dts.append(df)
    return pd.concat(dts)

def create_train_set(inputdir : str, outputdir: str , stock_list_not_to_use : np.array ,
                     number_of_stocks_to_use : int =None,
                     min_length : int = 150):
    '''
    Create train and validation data
    :param inputdir:
    :param outputdir:
    :param stock_list_not_to_use:
    :param min_length - minimal length of a stock
    :return:
    '''
    os.makedirs(outputdir, exist_ok=True)
    # get all stocks that can be trained on
    all_stocks = np.array([d for d in os.listdir(inputdir) if (os.path.isdir(os.path.join(all_stock_dir, d)) & (d not in stock_list_not_to_use)) ])
    stocks_to_use = all_stocks[np.random.permutation(len(all_stocks))]
    if number_of_stocks_to_use is None:
        number_of_stocks_to_use = len(stocks_to_use)
    else:
        stocks_to_use = stocks_to_use[:number_of_stocks_to_use]
        number_of_stocks_to_use = len(stocks_to_use)

    indices = np.random.permutation(number_of_stocks_to_use)
    # Define split sizes
    train_size = int(0.7 * number_of_stocks_to_use)
    val_size = number_of_stocks_to_use - train_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    train_df = dataset_preproc_conversion(inputdir,  stocks_to_use[train_indices], min_length)

    train_df.to_csv(os.path.join(outputdir, 'train_stocks.csv'))

    val_df = dataset_preproc_conversion(inputdir, stocks_to_use[val_indices], min_length)
    val_df.to_csv(os.path.join(outputdir, 'val_stocks.csv'))



if __name__ == "__main__":
    # datadir ='C:/Users\dadab\projects/algotrading\data/snp500'
    #
    # snp = pd.read_csv('C:/Users\dadab\projects/algotrading\data/tickers\sp500_stocks.csv')
    #
    # create_index("C:/Users\dadab\projects/algotrading\data/tickers", datadir,sorted(snp['Ticker'].values) ,
    #              filter_by_length = False ,  )
    # prerprocess_data(datadir)



    all_stock_dir = 'C:/Users/dadab/projects/algotrading/data/tickers'
    datadir ='C:/Users/dadab/projects/algotrading/data/training/dbsmall'
    #
    snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/tickers/sp500_stocks.csv')

    create_train_set(all_stock_dir, datadir,sorted(snp['Ticker'].values) , 40)

    #
    # create_index("C:/Users\dadab\projects/algotrading\data/tickers", datadir,sorted(snp['Ticker'].values) ,
    #              filter_by_length = False ,  )

