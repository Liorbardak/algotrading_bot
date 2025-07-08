import numpy as np
import pandas as pd
import os
from utils.visualizer import plot_ticker , plot_overall, holding_per_time
from utils.report_utils import  HtmlReport
def tradesim_report(tickers_df , complement_df , snp_df , trade_hist_df ,outputdir):
    '''
    Report of the trade metrics
    '''


    #  Clip to relevant dates
    trade_hist_df.Date = [pd.Timestamp(v) for v in trade_hist_df.Date.values]
    start_date = np.max([trade_hist_df.Date.min() , snp_df.Date.min()])
    end_date =  np.min([trade_hist_df.Date.max(), snp_df.Date.max()])
    snp_df = snp_df[(snp_df.Date >= start_date) &  (snp_df.Date <= end_date)]
    trade_hist_df = trade_hist_df[(trade_hist_df.Date >= start_date) &  (trade_hist_df.Date <= end_date)]
    tickers_df = tickers_df[(tickers_df.Date >= start_date) &  (tickers_df.Date <= end_date)]

    # Get tickers used
    all_tickers = list(set(complement_df.ticker))
    tickers_that_were_in_portfolio = [k for k in trade_hist_df.keys() if k in all_tickers]
    tickers_that_were_not_in_portfolio = set(all_tickers) - set(tickers_that_were_in_portfolio)





   # Performance per year
    start_of_year = start_date
    res = []
    while  (start_of_year <= end_date):
        end_of_year = start_of_year.replace(month=12, day=31)

        profit = trade_hist_df[(trade_hist_df.Date <= end_of_year)].total_value.values[-1] /     trade_hist_df[(trade_hist_df.Date >= start_of_year)].total_value.values[0]-1
        snp_profit = snp_df[(snp_df.Date <= end_of_year)].Close.values[-1] /     snp_df[(snp_df.Date >= start_of_year)].Close.values[0]-1
        res.append({'year': start_of_year.year, 'profit[%]' : np.round(profit*100,1), 'snp_profit[%]' : np.round(snp_profit*100,1)})
        start_of_year = start_of_year.replace(year=start_of_year.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    overall_profit = trade_hist_df[(trade_hist_df.Date == end_date)].total_value.values[0] /     trade_hist_df[(trade_hist_df.Date == start_date)].total_value.values[0] -1
    overall_snp_profit = snp_df[(snp_df.Date == end_date)].Close.values[0] /     snp_df[(snp_df.Date == start_date)].Close.values[0] -1
    res.append({'year': 'all', 'profit[%]' : np.round(overall_profit*100,1), 'snp_profit[%]' :  np.round(overall_snp_profit*100,1)})

    report = HtmlReport()
    report.add_df('Results' , pd.DataFrame(res))

    fig = plot_overall(snp_df, trade_hist_df)
    report.add_figure('bot vs snp', fig)

    # holdings per time
    fig = holding_per_time(trade_hist_df, tickers_that_were_in_portfolio)

    report.add_figure("holdings", fig)


    # add per ticker figures

    # draw tickers that were in portfolio first
    for ticker in   sorted(tickers_that_were_in_portfolio):
        fig = plot_ticker(ticker, tickers_df[tickers_df.ticker == ticker], complement_df[complement_df.ticker == ticker], trade_hist_df)
        report.add_figure(ticker, fig)
        import pylab as plt
        plt.close("all")

    # draw tickers that were not in portfolio first
    # for ticker in   tickers_that_were_not_in_portfolio:
    #     fig = plot_ticker(ticker, tickers_df[tickers_df.ticker == ticker], complement_df[complement_df.ticker == ticker], trade_hist_df)
    #     report.add_figure('ticker', fig)
    #     import pylab as plt
    #     plt.close("all")




    report.to_file(os.path.join(outputdir, 'report.html'))







