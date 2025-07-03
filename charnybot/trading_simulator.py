import pandas as pd
import os
from trade_policy import TradingPolicyBase
from data_loader import FinancialDataLoaderBase


def main():
    dataloader = FinancialDataLoaderBase()
    policy = TradingPolicyBase()

    tickers = ["ADM", "ADP"]
    stocks_data = dataloader.load_all_data(tickers ,get_average_stock = True )

    simulation_dates_range = stocks_data['actual_min_max_dates']
    reference_index = stocks_data['avg_df']

    policy.trade(simulation_dates_range , stocks_data['stocks_df'] ,  stocks_data['complement_df'] , reference_index )

    #for ticker in tickers:









if __name__ == "__main__":
    main()