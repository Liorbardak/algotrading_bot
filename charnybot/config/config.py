import json
import os

# Configuration paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data")
TICKERS_DIR = os.path.join(DATA_DIR, "tickers")
COMPLEMENT_RESULTS_DIR = os.path.join(DATA_DIR, "complements")

def get_config():
    '''
    Read the config file
    '''
    params = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),  'trading_config.json')))
    return params
