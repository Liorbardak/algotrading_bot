import pandas as pd
import os
from config.config import ConfigManager
from trade_policy import TradingPolicy
from data_loader import FinancialDataLoaderBase
from metrics import tradesim_report

class TrainingSimulator:
    """
    Main class for the training simulator that orchestrates:
    - Trading policy management
    - Data loading and preprocessing
    - Simulation execution
    - Results storage and visualization
    - Configuration management
    """

    def __init__(self, config):

        # Initialize components
        self.data_loader = FinancialDataLoaderBase(config)
        self.trade_policy = TradingPolicy.create(config.get_parameter("policy","policy_name") , config , default_index_name = 'snp')
    def run_training_simulation(self, start_date = None, end_date = None ,outputpath = None ):
        snp_df = self.data_loader.load_snp()
        stocks_df, complement_df, actual_min_max_dates, avg_df  = self.data_loader.load_all_data()

        self.trade_policy.trade(stocks_df, complement_df,snp_df, start_date= start_date,end_date=end_date,outputpath = outputpath  )

        if outputpath is not None:
            outfile = os.path.join(outputpath, 'trade_sim.csv')
            trade_hist_df = pd.read_csv(outfile)
            tradesim_report(stocks_df, complement_df, snp_df , trade_hist_df, outputpath)



def main(start_date = None, end_date = None, outputpath=None):
    config = ConfigManager()
    tradingsim = TrainingSimulator(config = config)
    tradingsim.run_training_simulation(start_date= start_date,end_date=end_date,outputpath=outputpath )

if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    outputpath = 'C:/Users/dadab/projects/algotrading/results/trading_sim/test4'

    main(start_date= start_date , end_date=end_date , outputpath=outputpath)