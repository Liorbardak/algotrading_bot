import os.path
from typing import List
import pandas as pd
import matplotlib

from basic_code.utils.report_utils import HtmlReport

# matplotlib.use('Qt5Agg')
# import pylab as plt

from bots import *
from visualization.visualize_trade import visualize_trade , visualize_all_bots


from basic_code.trading_simulations.tradesim import TradeSimSimple
import pickle



    
def run_trade_sim(datadir : str ,
                  results_dir: str,
                  trade_bots : List[BaseBot] , run_this_stock_only : str =None ,
                  fix_reference_index = False ,do_report: bool = True,
                  reference_key : str = 'close' #'4. close'
                 ):
    '''
    Run a simple trade simulation
    :param datadir:
    :param trade_bots:
    :param run_this_stock_only:
    :param fix_reference_index:
    :param do_report : save report per bot
    :param reference_key :  price to work with
    :return:
    '''
    os.makedirs(results_dir , exist_ok=True)
    # Read data
    stocks_df = pd.read_csv(os.path.join(datadir, 'all_stocks.csv'))    
    reference_index = pd.read_csv(os.path.join(datadir, 'reference_index.csv'))

    # Set the reference price to be used
    stocks_df['price'] = stocks_df[reference_key]
    reference_index['price'] = reference_index[reference_key]

    if fix_reference_index:
        # The alternative is to do nothing (=> the alternative price never changes)
        reference_index['price'] = 1.0

    if run_this_stock_only is not None:
        # Debug  - run only on one stock
        stocks_df = stocks_df[stocks_df['name'] == run_this_stock_only]

    #  Loop on all trade bot , simulate & report
    for trade_bot in trade_bots:

        if do_report:
            report = HtmlReport()
        else:
            report = None

        tradeSimulator = TradeSimSimple(algoBot=trade_bot)
        trade_info = tradeSimulator.run_trade_sim(stocks_df,  reference_index , report)

        pickle.dump(trade_info,  open(os.path.join(results_dir, f"res_{trade_bot._name}.pickle"), 'wb'))

        visualize_trade(trade_info , stocks_df, reference_index , report)
        if do_report:
            report.to_file(os.path.join(results_dir, f"res_{trade_bot._name}.html"))


    visualize_all_bots(datadir, results_dir, trade_bots , reference_key)






if __name__ == "__main__":
    #datadir = "C:\work\Algobot\data\INCY"
    datadir = "C:\work\data\snp500"
    results_dir =  "C:\work\data\/tradeRes\snp500"
    #run_trade_sim(datadir=datadir,results_dir=results_dir, trade_bots= [CharnyBot()] , run_this_stock_only='CMI')
    run_trade_sim(datadir=datadir, results_dir=results_dir, trade_bots=[CharnyBotBase(),CharnyBotV0(),macdWithRSIBot(), macdBot() , MACrossBot()])
    







