import json
import pandas as pd
import os
import numpy as np
import glob
import pylab as plt
from matplotlib.ticker import MaxNLocator
from soupsieve import closest
import re

def trade_sim(stockdir,  complement_dir , summary_name = 'summarizeCompliments.json',take_good_compliments = True):
    '''
    Simple trade simulation
    :return:
    '''

    # get ticker to run on and the average of all tickers
    tickers, all_stocks_average = get_relevant_tickers(complement_dir, stockdir,summary_name )

    all_stocks_average['Dates'] = pd.to_datetime(all_stocks_average.Date)

    mindate = all_stocks_average['Dates'].min()
    maxdate = all_stocks_average['Dates'].max()
    mindate = pd.to_datetime('2023-01-01')
    maxdate = pd.to_datetime('2024-12-30')
    all_res = []

    for ticker in sorted(tickers):
        print(f" ticker {ticker}")
        # Get stock price & some features
        stock_price = pd.read_csv(os.path.join(stockdir, ticker, 'stockPrice.csv'))
        stock_price['ma_200'] = stock_price['Close'].rolling(window=200).mean()
        stock_price['ma_150'] = stock_price['Close'].rolling(window=150).mean()
        ma_150_diff = np.diff(stock_price['ma_150'].values)
        stock_price['ma_150_slop'] = np.hstack((ma_150_diff[0], ma_150_diff))

        stock_dates = pd.to_datetime(stock_price.Date)

        # Get the complements
        #comps = json.load(open(os.path.join(complement_dir, ticker, "summarizeCompliments.json")))
        comps = json.load(open(os.path.join(complement_dir, ticker, summary_name)))

        prev_sell_date = stock_dates[0]

       # iterate on all analysit parties , decide when to buy
        for comp in comps:



            comp_date = (pd.Timestamp(comp['date'])).tz_localize(None)
            if comp_date < mindate or comp_date > maxdate:
                continue
            # fix formating problems
            for k in comp.keys():
                if (type(comp[k]) == list):
                    if (len(comp[k]) == 1):
                        comp[k] = comp[k][0]

            # Check  compliment condition to buy
            number_of_compliments = comp['number_of_analysts_comp_1'] + comp['number_of_analysts_comp_2'] + comp[
                    'number_of_analysts_comp_3']


            compliments_buy_indication = ((number_of_compliments >= 3) & (
                        number_of_compliments / (comp['total_number_of_analysts']+1e-6) > 0.5)) | (number_of_compliments >= 5)

            # Invert complements direction
            if take_good_compliments == False:
                compliments_buy_indication = not(compliments_buy_indication)

            if compliments_buy_indication == False:
                continue

            # Check for a valid buying date
            possible_buying_dates = stock_dates[stock_dates >= comp_date]
            closest_possible_date = possible_buying_dates.values[np.argmin(np.abs(possible_buying_dates - comp_date))]
            comp_ind = np.argmin(np.abs(stock_dates - closest_possible_date))
            # Get buy date - open price , one day after the complement
            buy_date = (comp_date + pd.Timedelta(days=1)).normalize()

            possible_buying_dates = stock_dates[stock_dates >= buy_date]
            closest_possible_to_buy = possible_buying_dates.values[np.argmin(np.abs(possible_buying_dates - buy_date))]

            if np.min(np.abs(stock_dates - buy_date)) > pd.Timedelta(days=3):
                print(f"no buying date for {ticker} at {buy_date}")

            buy_ind = np.argmin(np.abs(stock_dates - closest_possible_to_buy))
            buy_date = stock_dates[buy_ind]

            # Stock price related buy condition
            buy_cond1 = stock_price['Close'][comp_ind] > stock_price['ma_150'][comp_ind]
            buy_cond2 = stock_price['ma_150_slop'][comp_ind] > 0

            if (buy_cond1 & buy_cond2) == False:
                continue

            # Do not allow  buying before selling
            if (buy_date < prev_sell_date):
                continue

            # Buy @ open
            buy_price = stock_price.Open.values[buy_ind]


            # Sell condition

            prevday_ma = np.hstack([stock_price['ma_200'].values[0], stock_price['ma_200'].values[:-1], ])
            prevday_price = np.hstack([stock_price['Close'].values[0], stock_price['Close'].values[:-1]])


            # Sell condition - 2 days after price goes below 200ma
            sel_cond1 = (stock_price['ma_200'][buy_ind:] > stock_price['Close'][buy_ind:]) & (
                    prevday_ma[buy_ind:] > prevday_price[buy_ind:])
            sell_conds = np.where(sel_cond1)[0]

            if len(sell_conds) > 0:
                sellind = sell_conds[0] + buy_ind
                sell_date = stock_price['Date'][sellind]
                sell_price = stock_price['Close'][sellind]
                snp_sell_price = stock_price.snp_Close.values[sellind]

            else:
                # Take the last date
                sellind = len(stock_price) - 1
                sell_date = stock_price['Date'][sellind]
                sell_price = stock_price['Close'][sellind]
                snp_sell_price = stock_price.snp_Close.values[sellind]


            snp_buy_price = stock_price['snp_Close'][buy_ind]
            snp_sell_price = stock_price['snp_Close'][sellind]


            stocks_average_buy_price = all_stocks_average[all_stocks_average['Dates'] == buy_date].Open.values[0]
            stocks_average_sell_price = all_stocks_average[all_stocks_average['Dates'] == sell_date].Close.values[0]

            prev_sell_date = pd.to_datetime(sell_date)

            res = {'ticker': ticker, 'report_date': str(comp_date).split(' ')[0],
                   'buy_date': str(buy_date).split(' ')[0], 'buy_price(open)': np.round(buy_price, 3),
                   'sell_date': str(sell_date).split(' ')[0],
                   'sell_price(close)': np.round(sell_price, 3),
                   'profit': np.round(100 * (sell_price - buy_price) / buy_price, 1),
                   'snp_profit': np.round(100 * (snp_sell_price - snp_buy_price) / snp_buy_price, 1),
                   'average_stock_profit': np.round(
                       100 * (stocks_average_sell_price - stocks_average_buy_price) / stocks_average_buy_price, 1),
                   'n_analysts': comp['total_number_of_analysts'],
                   'n_comp': number_of_compliments,
                   'price / ma_150_at_buy (> 1)': np.round(sell_price / stock_price['ma_150'][comp_ind], 2),
                   'ma_150_slop_at_buy (> 0)': np.round(stock_price['ma_150_slop'][comp_ind], 2),
                   'price / ma_200_at_sell (< 1)': np.round(
                       stock_price['Close'][sellind] / stock_price['ma_200'][sellind], 4),
                   'price / ma_200_1day_before_sell (< 1)': np.round(
                       stock_price['Close'][sellind - 1] / stock_price['ma_200'][sellind - 1], 4),
                   'price / ma_200_2day_before_sell (> 1)': np.round(
                       stock_price['Close'][sellind - 2] / stock_price['ma_200'][sellind - 2], 4),

                   }

            # Sell due to last date
            if sell_date == stock_price['Date'].max():
                res['price / ma_200_at_sell'] = 'sold @ last date'
                res['price / ma_200_day_before_sell'] = 'sold @ last date'
                res['price / ma_200_2day_before_sell'] = 'sold @ last date'

            all_res.append(res)

    return pd.DataFrame(all_res)



def get_relevant_tickers(complement_dir , stockdir , summary_name ):
    '''
    Get the relevant tickers with compliments - To be revised
    :param complement_dir:
    :return:
    '''
    relevant_tickers = []
    mindate = None
    maxdate = None

    # comphrase= 'parsed_*gpt41*.json'
    # comphrase= r'^parsed_\d{4}_[1-4]_results'

    for ticker in os.listdir(complement_dir):
        if os.path.isdir(os.path.join(complement_dir, ticker)) == False:
            continue

        if os.path.isfile(os.path.join(complement_dir, ticker,summary_name)):
            relevant_tickers.append(ticker)
            if os.path.isfile(os.path.join(stockdir, ticker, 'stockPrice.csv')) == False:
                print(f"no stock {ticker}")
            else:
                df = pd.read_csv(os.path.join(stockdir, ticker, 'stockPrice.csv'))
                # Check for dates range valid for all stocks
                if mindate == None:
                    mindate =  pd.to_datetime(df['Date']).min()
                    maxdate =  pd.to_datetime(df['Date']).max()
                else:
                    mindate = np.max([mindate,  pd.to_datetime(df['Date']).min()])
                    maxdate = np.min([maxdate,  pd.to_datetime(df['Date']).max()])

        else:
            pass
            #print(f"no comp {ticker}")
    if len(relevant_tickers) == 0:
        print('NO VALID TICKERS')
        return None, None

    # Get "average" stock
    df = pd.read_csv(os.path.join(stockdir, relevant_tickers[0], 'stockPrice.csv'))
    all_df = df[( pd.to_datetime(df.Date) >= mindate) & ( pd.to_datetime(df.Date) <= maxdate)].reset_index()
    for ticker in relevant_tickers[1:]:
        df = pd.read_csv(os.path.join(stockdir, ticker, 'stockPrice.csv'))
        df =  df[( pd.to_datetime(df.Date) >= mindate) & ( pd.to_datetime(df.Date) <= maxdate)].reset_index()
        for k in [k for k in df.keys() if not k in ['Date','index']]:
            all_df[k] = all_df[k].values + df[k].values

    for k in [k for k in df.keys() if not k in ['Date','index']]:
        all_df[k] = all_df[k].values / len(relevant_tickers)

    return relevant_tickers, all_df

def summerize_comment(complement_dir ,  comphrase =  r'^parsed_\d{4}_[1-4]_gpt41_2023' , summary_name = 'summarizeCompliments_gpt41_2023.json'):
    '''
    Summerize comments
    '''
    #comphrase =  r'^parsed_\d{4}_[1-4]_gpt41_2023'
    #comphrase = r'^parsed_\d{4}_[1-4]_results'
    for ticker in [ticker for ticker in os.listdir(complement_dir) if os.path.isdir(os.path.join(complement_dir, ticker))]:
        compfiles  =[
            filename for filename in os.listdir(os.path.join(complement_dir, ticker))
            if re.match(comphrase, filename)
        ]
        if len(compfiles) == 0:
            continue
            print(f"no comphrase {ticker}")
        else:
            summerize_all = []
            for compfile in compfiles:
                comps = json.load(open(os.path.join(complement_dir, ticker,compfile)))
                compnum = {0:0 , 1:0 , 2:0 , 3:0}
                for comp in comps:
                    compnum[comp['level']] += 1
                summerize_all.append({'date': comps[0]['date'],
                                      'total_number_of_analysts':sum(compnum.values()),
                                      'number_of_analysts_comp_1': compnum[1],
                                      'number_of_analysts_comp_2': compnum[2],
                                      'number_of_analysts_comp_3': compnum[3]
                })
            with open(os.path.join(complement_dir, ticker,summary_name), 'w') as f:
                json.dump(summerize_all, f, indent=4)  # indent=4 makes i



if __name__ == "__main__":

    complement_dir = '../data/gpt41_pretest'
    #complement_dir = '../data/analysis'
    stockdir = '../data/tickers'
    #summerize_comment(complement_dir)

    df = trade_sim(stockdir, complement_dir , summary_name = 'summarizeCompliments_gpt41_2023.json' )
    print("good compliments buy points:" )
    print(f'profit[%] {np.mean(df.profit):.2f}  snp profit[%] {np.mean(df.snp_profit):.2f} ,average stock profit[%] {np.mean(df.average_stock_profit):.2f}' )
