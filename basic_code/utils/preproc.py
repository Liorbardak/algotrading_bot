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
            df.loc[df.name == stock_name, k] = df[df.name == stock_name][k] * normFact

    # average on all stocks per time
    res = []
    for date, df_date in df.groupby('Date'):
        r = {'Date': date, 'name': 'average'}
        for k in keys_to_average:
            r[k] = df_date[k].mean()
        res.append(r)
    return pd.DataFrame(res)




if __name__ == "__main__":
    #dataindir = "C:\work\data\snp100"

    prerprocess_data('C:\work\data\snp500_filtered')

    # snp = pd.read_csv('C:\work\data\/tickers\sp500_stocks.csv')
    # create_index('C:\work\data\/tickers', 'C:\work\data\snp500_filtered',sorted(snp['Ticker'].values) , filter_by_length = True)



