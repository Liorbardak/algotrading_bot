from typing import Dict , List
import os
import numpy as np
import pandas as pd
import pickle
import pylab as plt
from basic_code.utils.report_utils import HtmlReport
def display_prediction(resfile:str ,datadir : str , pred_len = 20 ):
    predictions = pd.read_csv(resfile)
    for stock_name,prediction in predictions.groupby('stock'):
        df = pd.read_excel(os.path.join(datadir, stock_name, 'stockPrice.xlsx'), engine='openpyxl')
        plt.plot(df['close'].values)
        plt.title(stock_name)
        for index in prediction.time_index.values[::20]:
              row = prediction[prediction.time_index == index]
              predvector = np.zeros(pred_len,)
              for p in np.arange(pred_len):
                  predvector[p] = row['pred' + str(p)]
              plt.plot(np.arange(index, index+pred_len) , predvector)
        plt.show()






if __name__ == "__main__":
    resfile = 'C:/Users/dadab/projects/algotrading/data/training/predictions/predictions.csv'
    datadir  = 'C:/Users/dadab/projects/algotrading\data/tickers'
    display_prediction(resfile,datadir )