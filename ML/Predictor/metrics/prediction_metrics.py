import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import pylab as plt
from typing import List

def run_prediction_metrics(dbname: str, predictors: str, results_dir : str, prediction_times : List ):
    for predictor in predictors:
        df = pd.read_csv(os.path.join(results_dir, f"{dbname}_{predictor}"))
        for ticker, sdf in  df.groupby('name'):



if __name__ == "__main__":
    dbname = 'snp_v0'
    predictors = ['simp_tf']
    prediction_times = [2,5,10]
    results_dir = f"C:/Users/dadab/projects/algotrading/results"
    run_prediction_metrics(dbname, predictors, results_dir, prediction_times)
