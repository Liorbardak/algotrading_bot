import ctypes
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path
from typing import List
import pandas as pd
import matplotlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from basic_code.utils.report_utils import HtmlReport

from bots import *
from visualization.visualize_trade import visualize_trade , visualize_all_bots


from basic_code.trading_simulations.tradesim import TradeSimSimple


import pickle


def run_bot(inputs : List , trade_bot : BaseBot):
    '''
    Run a single bot simulation
    :param inputs:
    :param trade_bot:
    :return:
    '''
    [results_dir, stocks_df, reference_index, do_report] = inputs
    if do_report:
        report = HtmlReport()
    else:
        report = None

    tradeSimulator = TradeSimSimple(algoBot=trade_bot)
    trade_info = tradeSimulator.run_trade_sim(stocks_df, reference_index, report)

    pickle.dump(trade_info, open(os.path.join(results_dir, f"res_{trade_bot._name}.pickle"), 'wb'))

    #visualize_trade(trade_info, stocks_df, reference_index, report)
    if do_report:
        report.to_file(os.path.join(results_dir, f"res_{trade_bot._name}.html"))


def run_trade_sim(datadir : str ,
                  results_dir: str,
                  trade_bots : List[BaseBot] , run_this_stock_only : str =None ,
                  fix_reference_index = False ,do_report: bool = False,
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
    import ctypes
    import time

    # Prevent sleep
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


    os.makedirs(results_dir , exist_ok=True)
    # Read data
    stocks_df = pd.read_csv(os.path.join(datadir, 'all_stocks.csv'))    
    reference_index = pd.read_csv(os.path.join(datadir, 'reference_index.csv'))

    if fix_reference_index:
        # The alternative is to do nothing (=> the alternative price never changes)
        reference_index['close'] = 1.0

    if run_this_stock_only is not None:
        # Debug  - run only on one stock
        stocks_df = stocks_df[stocks_df['name'] == run_this_stock_only]

    #  Loop on all trade bot , simulate & report
    if len(trade_bots) == 1:
        for trade_bot in trade_bots:
            run_bot([results_dir,stocks_df ,  reference_index, do_report], trade_bot)
    else:
        # Parallel
        partial_task = partial(run_bot, [results_dir,stocks_df ,  reference_index, do_report])

        with ThreadPoolExecutor(max_workers=len(trade_bots)) as executor:
            results = list(executor.map(partial_task, trade_bots))


    # Visualize all bots
    visualize_all_bots(datadir, results_dir, trade_bots)






if __name__ == "__main__":
    datadir = 'C:/Users/dadab/projects/algotrading/data/snp500'
    results_dir =  "C:/Users/dadab/projects/algotrading/results/playground"

    #
    # run_trade_sim(datadir=datadir, results_dir=results_dir,
    #               trade_bots=[CharnyBotV2()], do_report=True)

    run_trade_sim(datadir=datadir, results_dir=results_dir,
                  trade_bots=[CharnyBotV1() , CharnyBotV2()], do_report=False)
    #
    # run_trade_sim(datadir=datadir, results_dir=results_dir,
    #               trade_bots=[CharnyBotPlayground()], do_report=True,  run_this_stock_only='ADI')
