import pandas as pd
def add_pred():
    #add ma20 prediction to "standard" prediction
    df = pd.read_csv("C:/Users/dadab/projects/algotrading/results/inference/snp_v5_lstm1/ticker_data_with_prediction.csv")
    df_w_pred_ma = pd.read_csv(
        "C:/Users/dadab/projects/algotrading/results/inference/snp_v5_ma20_lstm1/ticker_data_with_prediction.csv")
    df['pred5_ma20'] = df_w_pred_ma['pred5']
   # df.dropna(inplace=True)
    df.to_csv('C:/Users/dadab/projects/algotrading/results/inference/snp_v5_lstm1/ticker_data_with_prediction.csv')

if __name__ == "__main__":
    add_pred()