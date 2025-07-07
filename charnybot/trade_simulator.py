from pandas.core.indexes.api import default_index

from config.config import ConfigManager
from trade_policy import TradingPolicy
from data_loader import FinancialDataLoaderBase

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
    def run_training_simulation(self, start_date = None, end_date = None):
        snp_df = self.data_loader.load_snp()
        stocks_df, complement_df, actual_min_max_dates, avg_df  = self.data_loader.load_all_data()

        self.trade_policy.trade(stocks_df, complement_df,snp_df, start_date= start_date,end_date=end_date )



def main(start_date = None, end_date = None):
    config = ConfigManager()
    tradingsim = TrainingSimulator(config = config)
    tradingsim.run_training_simulation(start_date= start_date,end_date=end_date )

if __name__ == "__main__":
    start_date = '2020-10-05'
    main(start_date= start_date)