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
        self.trade_policy = TradingPolicy.create(config.get_parameter("policy","policy_name") , config)
    def run_training_simulation(self):
        snp_df = self.data_loader.load_snp()
        stocks_df, complement_df, actual_min_max_dates, avg_df  = self.data_loader.load_all_data()

        self.trade_policy.trade(stocks_df, complement_df,snp_df )



def main():
    config = ConfigManager()
    tradingsim = TrainingSimulator(config = config)
    tradingsim.run_training_simulation()

if __name__ == "__main__":
    main()