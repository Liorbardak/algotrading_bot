import json
import pandas as pd
import os
import numpy as np
import pylab as plt
from matplotlib.ticker import MaxNLocator
from soupsieve import closest


def trade_sim(stockdir , df_comp ,  df_snp , all_stocks_average , ticker_names):

    df_snp['Dates'] = pd.to_datetime(df_snp.Date)
    all_stocks_average['Dates'] = pd.to_datetime(all_stocks_average.Date)

    all_res = []

    for ticker in sorted(ticker_names):

        stock_price = pd.read_csv(os.path.join(stockdir, ticker + '.csv'))
        stock_price['ma_200'] = stock_price['Close'].rolling(window=200).mean()
        stock_price['ma_150'] = stock_price['Close'].rolling(window=150).mean()
        ma_150_diff = np.diff( stock_price['ma_150'].values)
        stock_price['ma_150_slop'] = np.hstack(( ma_150_diff[0],  ma_150_diff))

        stock_dates = pd.to_datetime(stock_price.Date)

        df = df_comp[df_comp.ticker == ticker]
        prev_sell_date = stock_dates[0]

        # iterate on all analysit parties , decide when to buy
        for ri, r in df.iterrows():
            comp_date = (pd.Timestamp(r['date'])).tz_localize(None)


            possible_buying_dates = stock_dates[stock_dates >= comp_date]
            closest_possible_date  = possible_buying_dates.values[np.argmin(np.abs(possible_buying_dates - comp_date))]
            comp_ind = np.argmin(np.abs(stock_dates - closest_possible_date))

            # Check buying slop



            # Get buy date - open price , one day after the complement
            buy_date  =  (comp_date + pd.Timedelta(days=1)).normalize()

            possible_buying_dates = stock_dates[stock_dates >= buy_date]
            closest_possible_to_buy  = possible_buying_dates.values[np.argmin(np.abs(possible_buying_dates - buy_date))]

            if np.min(np.abs(stock_dates - buy_date)) > pd.Timedelta(days=3):
                print(f"no buying date for {ticker} at {buy_date}")



            buy_ind = np.argmin(np.abs(stock_dates - closest_possible_to_buy))
            buy_date = stock_dates[buy_ind]

            # Buy condition
            buy_cond1 = stock_price['Close'][comp_ind] > stock_price['ma_150'][comp_ind]
            buy_cond2 = stock_price['ma_150_slop'][comp_ind] > 0
            if (buy_cond1 & buy_cond2) == False:
                continue
            # Do not allow  buying before selling
            if (buy_date < prev_sell_date):
                continue


            # Buy @ open
            buy_price = stock_price.Open.values[buy_ind]


            #snp_buy_price = stock_price.snp_Open.values[buy_ind]

            # Sell condition

            prevday_ma = np.hstack([stock_price['ma_200'].values[0], stock_price['ma_200'].values[:-1], ])
            prevday_price = np.hstack([stock_price['Close'].values[0], stock_price['Close'].values[:-1]])

            # nextday_ma = np.hstack([stock_price['ma_200'].values[1:], stock_price['ma_200'].values[-1]])
            # nextday_price = np.hstack([stock_price['Close'].values[1:], stock_price['Close'].values[-1]])



            # fig, ax = plt.subplots()

            # ax.plot(stock_price.Date, stock_price['ma_200'])
            # ax.plot(stock_price.Date, stock_price['Close'])
            # # Set maximum number of x-ticks
            # ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
            # ax.set_title(f'ticker {ticker} date {buy_date}')
            # plt.show()

            # Sell condition - 2 days after price goes below 200ma
            sel_cond1 = (stock_price['ma_200'][buy_ind:] > stock_price['Close'][buy_ind:]) & (
                        prevday_ma[buy_ind:] > prevday_price[buy_ind:])
            sell_conds = np.where(sel_cond1)[0]

            #or if price goes 10 %below ma200
            # sel_cond2 = (stock_price['ma_200'][buy_ind:] > 1.10 * stock_price['Close'][buy_ind:])
            # sell_conds = np.where(sel_cond1 | sel_cond2)[0]

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


            try:
                snp_buy_price = df_snp[df_snp['Dates'] == buy_date].Open.values[0]
                snp_sell_price = df_snp[df_snp['Dates'] == sell_date].Close.values[0]
            except:
                print(f'no valid snp {ticker} ')
                continue

            stocks_average_buy_price = all_stocks_average[all_stocks_average['Dates'] == buy_date].Open.values[0]
            stocks_average_sell_price = all_stocks_average[all_stocks_average['Dates'] == sell_date].Close.values[0]


            prev_sell_date = pd.to_datetime(sell_date)



            res = {'ticker': ticker,'report_date': str(comp_date).split(' ')[0] ,  'buy_date': str(buy_date).split(' ')[0], 'buy_price(open)': np.round(buy_price, 3),
                   'sell_date': str(sell_date).split(' ')[0],
                   'sell_price(close)': np.round(sell_price, 3),
                   'profit': np.round(100 * (sell_price - buy_price) / buy_price, 1),
                   'snp_profit': np.round(100 * (snp_sell_price - snp_buy_price) / snp_buy_price, 1),
                   'average_stock_profit': np.round(100 * (stocks_average_sell_price - stocks_average_buy_price) / stocks_average_buy_price, 1),
                   'n_analysts': r.total_number_of_analysts,
                   'n_comp': r.number_of_analysts_comp_1 + r.number_of_analysts_comp_2 + r.number_of_analysts_comp_3,
                   'price / ma_150_at_buy (> 1)':  np.round(sell_price / stock_price['ma_150'][comp_ind],2),
                   'ma_150_slop_at_buy (> 0)':   np.round(stock_price['ma_150_slop'][comp_ind],2),
                   'price / ma_200_at_sell (< 1)':  np.round( stock_price['Close'][sellind] / stock_price['ma_200'][sellind],4),
                   'price / ma_200_1day_before_sell (< 1)':  np.round( stock_price['Close'][sellind-1] / stock_price['ma_200'][sellind-1],4),
                   'price / ma_200_2day_before_sell (> 1)':  np.round( stock_price['Close'][sellind-2] / stock_price['ma_200'][sellind - 2],4),

                   }


            # Sell due to last date
            if sell_date == stock_price['Date'].max():
                res['price / ma_200_at_sell'] = 'sold @ last date'
                res['price / ma_200_day_before_sell'] = 'sold @ last date'
                res['price / ma_200_2day_before_sell'] = 'sold @ last date'

            all_res.append(res)


    return pd.DataFrame(all_res)

if __name__ == "__main__":

    inpath = '../data/stocks_quality/analysis/analysis'
    stockdir = 'C:/Users/dadab/projects/algotrading/data/snp500_yahoo'
    df_comp = pd.read_csv(inpath + '/ComplimentsInfo.csv')

    df_snp = pd.read_csv(stockdir + '/snp500_yahoo.csv')



    number_of_compliments = df_comp.number_of_analysts_comp_1 + df_comp.number_of_analysts_comp_2 + df_comp.number_of_analysts_comp_3
    #number_of_compliments =  (df.number_of_analysts_comp_3 + df.number_of_analysts_comp_2 + df.number_of_analysts_comp_1) / ( df.total_number_of_analysts + 1)
    df_comp['number_of_compliments'] = number_of_compliments

    good_stock_comp = ((number_of_compliments >= 3 ) & (number_of_compliments / df_comp.total_number_of_analysts  > 0.5)) | (number_of_compliments >= 5 )
    good_stock_val  = ((df_comp['Close_0'] > df_comp['ma_150']) & (df_comp['ma_150_slop'] > 0))

    good_stock = good_stock_comp & good_stock_val
    df_comp['good_stock'] = good_stock

    ticker_names = list(set(df_comp.ticker))
    ticker_names = list(set(df_comp['ticker']) - set(['ADMA']))
    #ticker_names = ['ADBE' , 'ADI', 'ADMA' , 'AMAT' , 'ANDE' ,'ANET' , 'AVGO' , 'AXON' , 'BCRX' , 'BMY', 'BSX' ,'CDNS' , 'CELH' ,  'CL', 'CLBT' , 'CORT', 'CPRX' , 'CRM' ,'CRWD' ,'CYBR','DCTH','DHR', 'EW']
    #ticker_names = ['ANDE']
    #ticker_names = ['CRM']

    # Create reference index
    all_stocks_average = pd.read_csv(os.path.join(stockdir, ticker_names[0] + '.csv'))
    for ticker in ticker_names[1:]:
        stock_price = pd.read_csv(os.path.join(stockdir, ticker + '.csv'))
        all_stocks_average.Close = all_stocks_average.Close + stock_price.Close
        all_stocks_average.Open = all_stocks_average.Open + stock_price.Open
    all_stocks_average.Close = all_stocks_average.Close/ len(ticker_names)
    all_stocks_average.Open = all_stocks_average.Open / len(ticker_names)



    df = trade_sim(stockdir, df_comp[df_comp['good_stock'] == True], df_snp , all_stocks_average , ticker_names)
    df.to_csv(inpath + '/good_stock.csv')
    print("good compliments buy points:" )
    print(f'profit[%] {np.mean(df.profit):.2f}  snp profit[%] {np.mean(df.snp_profit):.2f} ,average stock profit[%] {np.mean(df.average_stock_profit):.2f}' )

    # df = trade_sim(stockdir, df_comp[df_comp['good_stock'] == False], df_snp , all_stocks_average , ticker_names)
    # df.to_csv(inpath + '/bad_stock.csv')
    # print("bad compliments buy points:" )
    # print(f'profit[%] {np.mean(df.profit):.2f}  snp profit[%] {np.mean(df.snp_profit):.2f} ,average stock profit[%] {np.mean(df.average_stock_profit):.2f}' )
