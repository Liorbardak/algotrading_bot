import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rls_predictor(signal, filter_order=4, delta=0.01, lam=0.99,pred=1):
    """
    RLS adaptive filter for signal prediction.

    Parameters:
        signal (np.array): Input signal
        filter_order (int): Number of taps (order) of the filter
        delta (float): Initialization value for P matrix
        lam (float): Forgetting factor (0 < lam ≤ 1)

    Returns:
        y_pred (np.array): Predicted output
        e (np.array): Prediction error
    """
    n_samples = len(signal)
    w = np.zeros(filter_order)
    P = (1 / delta) * np.eye(filter_order)

    y_pred = np.full(n_samples, np.nan)
    e = np.zeros(n_samples)

    for n in range(filter_order, n_samples-pred):
        x = signal[n - filter_order:n][::-1]  # input vector (reversed)
        y = np.dot(w, x)  # predicted value
        error = signal[n+pred] - y

        Pi_x = np.dot(P, x)
        k = Pi_x / (lam + np.dot(x, Pi_x))  # gain vector
        w = w + k * error  # weight update
        P = (P - np.outer(k, Pi_x)) / lam  # covariance update

        y_pred[n] = y
        e[n] = error

    y_pred[:filter_order+1] = np.nan
    return y_pred, e



def run_predictor(datadir: str, outputdir : str, pred_len : int  = 15):
    '''
    Simple rls predictor
    :param datadir:
    :param pred_len:
    :return:
    '''
    os.makedirs(outputdir, exist_ok=True)

    # Get the original stock prices
    df = pd.read_csv(os.path.join(datadir, 'test_df_orig.csv'))
    for t in np.arange(pred_len):
        df['pred' + str(t)] = np.nan

    for ticker, sdf in df.groupby('ticker'):
        x= sdf.close.values
        for t in np.arange(pred_len):
            y_pred, e =rls_predictor(x , pred = t )
            df.loc[sdf.index, 'pred' + str(t)] = y_pred

    df.to_csv(os.path.join(outputdir, 'ticker_data_with_prediction.csv'))

if __name__ == "__main__":
    params = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'training_config.json')))

    ddname = params['db']
    datadir = f'C:/Users/dadab/projects/algotrading/data/training/{ddname}/'
    outdir = f"C:/Users/dadab/projects/algotrading/results/{ddname}_rls"
    run_predictor(datadir, outdir)


