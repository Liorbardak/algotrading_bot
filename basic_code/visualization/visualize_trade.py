from typing import Dict , List
import os
import pandas as pd
import pickle

from basic_code.utils.report_utils import HtmlReport


def visualize_trade(trade_info : Dict , stocks_df : pd.DataFrame ,  reference_index : pd.DataFrame , report : "HtmlReport" = None):
    '''
    Visualize bot results
    :param trade_info:
    :param stocks_df:
    :param reference_index:
    :param report:
    :return:
    '''
    import matplotlib
    matplotlib.use('Qt5Agg')
    import pylab as plt
    fig = plt.figure()
    plt.plot(trade_info['total_value'] / trade_info['total_value'][0]*100 , label = 'trade')
    plt.plot(reference_index['price'], label = 'reference index ')
    plt.title('overall balance')
    plt.legend()
    plt.ylabel('price')
    plt.ylabel('day')
    if report:
        report.add_figure('overall balance' , fig)
    plt.close("all")
    return

    # for si, (stock_name, stock_df) in enumerate(stocks_df.groupby('name', sort=True)):
    #     stock_df = stock_df.reset_index()
    #
    #     fig, ax1 = plt.subplots()
    #
    #     ax2 = ax1.twinx()
    #     ax1.plot(trade_info['stocks_per_share'][si], 'g-')
    #     ax2.plot(stock_df.price / stock_df.price.values[0], 'b-')
    #     ax1.set_ylabel('number_of_stocks', color='g')
    #     ax2.set_ylabel('price', color='b')
    #
    #     plt.title(f' {stock_name}')
    #
    #     if report:
    #         report.add_figure(stock_name, fig)
    #     plt.close("all")


def visualize_all_bots(datadir: str,
                    results_dir: str,
                    trade_bots: List ,
                    reference_key : str = 'close'
                  ):
    '''
    Visualize all bots
    :param datadir
    :param results_dir:
    :param trade_bots:
    :param reference_key
    :return:
    '''
    # Read data
    reference_index = pd.read_csv(os.path.join(datadir, 'reference_index.csv'))
    reference_index['price'] = reference_index[reference_key]


    report = HtmlReport()
    import matplotlib
    matplotlib.use('Qt5Agg')
    import pylab as plt
    fig = plt.figure()
    plt.plot(reference_index['price'], label = 'reference index ')
    for trade_bot in trade_bots:
        trade_info = pickle.load(open(os.path.join(results_dir, f"res_{trade_bot._name}.pickle"), 'rb'))
        plt.plot(trade_info['total_value'] / trade_info['total_value'][0] * 100, label=f"{trade_bot._name}")
    plt.title('overall balance')
    plt.legend()
    plt.ylabel('price')
    plt.ylabel('day')
    report.add_figure('overall balance', fig)
    report.to_file(os.path.join(results_dir, f"res_all_bots.html"))
