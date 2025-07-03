import json
import pandas as pd
import os
import numpy as np
import ta

def read_complements(datadir, stockdir):
    comp_all = []


    for ticker in  [name for name in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, name))]:
        if(os.path.isfile(os.path.join(datadir, ticker, "summarizeCompliments.json"))):
            print(ticker)
            #load general data and add it to the complement
            #stockana = pd.read_excel(os.path.join(datadir, ticker, 'stocksAnalysis.xlsx'), engine='openpyxl')


            comps = json.load(open(os.path.join(datadir, ticker, "summarizeCompliments.json")))
            for comp in comps:

                print(comp['date'])
                comp_date = pd.to_datetime(comp['date']).tz_localize(None)

                # if np.min(np.abs(stockana['reportedDate'] - comp_date)) > pd.Timedelta(days=90):
                #     print(f"no stocksAnalysis for {ticker} at {comp['date']}")
                #     continue

                stock_price = pd.read_csv(os.path.join(stockdir,ticker + '.csv'))
                stock_dates =pd.to_datetime(stock_price['Date'])
                if (np.min(np.abs(stock_dates+ pd.Timedelta(days=200) - comp_date))   > pd.Timedelta(days=4))  |  (np.min(np.abs(stock_dates+ pd.Timedelta(days=0) - comp_date))   > pd.Timedelta(days=4)):
                    print(f"no stocksAnalysis for {ticker} at {comp['date']}")
                    continue

                for k in comp.keys():
                    if (type(comp[k]) == list):
                        if (len(comp[k]) == 1):
                            comp[k] = comp[k][0]

                comp['ticker'] = ticker

                # Add next time the price goes under MA200
                ma_200 = stock_price['Close'].rolling(window=200).mean()
                ma_150 = stock_price['Close'].rolling(window=150).mean()
                ma_150_diff = np.diff(ma_150.values)
                ma_150_slop = np.hstack((ma_150_diff[0], ma_150_diff))

                closest_ind = np.argmin(np.abs(stock_dates - comp_date))

                if np.isnan(ma_200[closest_ind]):
                    # ma_200 does not exist
                    continue
                comp['ma_200'] = ma_200[closest_ind]
                comp['ma_150'] = ma_150[closest_ind]
                comp['ma_150_slop'] = ma_150_slop[closest_ind]

                # Add the future price
                for future in [0,1, 10, 50, 100 , 200]:
                    closest_ind = np.argmin(np.abs(stock_dates + pd.Timedelta(days=future) - comp_date))
                    comp['Close_' + str(future)] = stock_price.iloc[closest_ind].Close

                comp_all.append(comp)
    return pd.DataFrame.from_dict(comp_all)





if __name__ == "__main__":
    inpath = 'C:/Users/dadab/projects/algotrading/data/stocks_quality/analysis/analysis'
    stockdir = 'C:/Users/dadab/projects/algotrading/data/snp500_yahoo'
    # stock_price = pd.read_excel(os.path.join(stockdir, 'CLBT', 'stockPrice.xlsx'))
    # plt.plot(stock_price['Date'] , stock_price['close'])
    # plt.title('CLBT')
    # plt.show()


    df = read_complements(inpath ,stockdir)
    df.to_csv(inpath + '/ComplimentsInfo.csv')