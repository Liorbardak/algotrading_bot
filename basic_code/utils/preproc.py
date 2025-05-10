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
    - Rename data cols
    - Save the reference index and the common stocks data frame
    :param datadir: input directory
    :param minLengthtoUse:  Minimal number of dates in a stock file directory
    '''


    dirnames = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
    dfs = []
    for dirname in dirnames:
        filename = os.path.join(datadir, dirname, 'stockPrice.xlsx')
        df = pd.read_excel(filename, engine='openpyxl')


        if(len(df)  < minLengthtoUse):
            # Not enough information
            continue


        df['ticker'] = dirname


        # Rename
        df.rename(
            columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', 'close': 'close', '5. volume': 'volume',
                     'Date': 'date'},
            inplace=True)

        df = df[['date','open', 'close', 'high', 'low', 'volume', 'ticker']]

        dfs.append(df)

        print(f'{dirname} from {np.min(df.date)}  to  {np.max(df.date)}  {len(df)}')

    df_all = pd.concat(dfs)

    # Take dates that has all stocks information
    Nstocks = len(set(df_all.ticker))
    filtered_df = []

    for date, df in df_all.groupby('date'):
        if(len(df) ==Nstocks):
            filtered_df.append(df)



    df_all = pd.concat(filtered_df).reset_index()

    if (len(set(df_all.date)) < minLengthtoUse):
        print('Error : no enough dates')
        return

    print(f' preroc {datadir} #stocks {len(set(df_all.ticker))} #dates {len(set(df_all.date))}  from {np.min(df_all.date)} to {np.max(df_all.date)} ')
    # Save data
    df_all.to_csv(os.path.join(datadir, 'all_stocks.csv'))
    # Get & Save  average index
    avgdata = get_average_stock(df_all)
    avgdata.to_csv(os.path.join(datadir, 'reference_index.csv'))



def create_index(inputdir : str, outputdir: str , stock_list : np.array, filter_by_length : bool = False , referenceDates = None):
    '''
    Create index Copy part of the stocks to output directory
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
                    referenceDates = df.date.values
                    shutil.copytree(os.path.join(inputdir, sname), os.path.join(outputdir, sname))
                    print(f'{nfiles} copy {sname} {len( df.date.values)}')
                    nfiles += 1

                else:
                    if len(set(referenceDates)- set(df.date.values)) == 0:
                        print(f'{nfiles} copy {sname}  {len( df.date.values)}')
                        nfiles += 1
                        shutil.copytree(os.path.join(inputdir, sname), os.path.join(outputdir, sname))
                    else:
                        print(f'{sname} does not fit  {len( df.date.values)}')
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
    keys_to_average =  ['open', 'high', 'low',reference_key , 'volume']

    # Normalize
    df = dfi.copy()
    refData = np.min(df.date)
    ticker = set(df.ticker)
    for stock_name in ticker:
       # normalize so first closing price will be 100
       normFact = 100 / df[(df.ticker == stock_name) & (df.date == refData)][reference_key].values[0]
       for k in keys_to_average:
           df[k] = df[k].astype(float)  # convert entire column if appropriate
           df.loc[df.ticker == stock_name, k] = df[df.ticker == stock_name][k] * normFact
           # df.loc[df.ticker == stock_name, k] = df[df.ticker == stock_name][k].astype(float) * normFact

    # average on all stocks per time
    res = []
    for date, df_date in df.groupby('date'):
        r = {'date': date, 'ticker': 'average'}
        for k in keys_to_average:
            r[k] = df_date[k].mean()
        res.append(r)
    return pd.DataFrame(res)

if __name__ == "__main__":
    np.random.seed(42)
    datadir ='C:/Users/dadab/projects/algotrading/data/snp500'

    snp = pd.read_csv('C:/Users/dadab/projects/algotrading/data/tickers/sp500_stocks.csv')

    # create_index("C:/Users\dadab\projects/algotrading\data/tickers", datadir,sorted(snp['Ticker'].values) ,
    #              filter_by_length = False ,  )
    prerprocess_data(datadir)




