import sys
import os
import numpy as np
import pandas as pd
import pylab as plt
import json
from typing import List
import seaborn as sns
from basic_code.utils.report_utils import HtmlReport
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ML.Predictor.config.config import get_config

def run_prediction_metrics(dbname: str, predictors: str, results_dir : str, outputdir : str , prediction_times : List  ):

    # Combine all predictors data
    df_with_gt = []
    for predictor in predictors:
        df = pd.read_csv(os.path.join(results_dir, f"{dbname}_{predictor}", "ticker_data_with_prediction.csv" ))
        df['predictor'] = predictor
        df_with_gt.append(df)
    df_with_gt = pd.concat(df_with_gt)
    df_with_gt.reset_index(inplace=True)

    for t in sorted(prediction_times):
        df_with_gt[f"pred_gt{t}"] = np.nan
        df_with_gt[f"pred_err{t}"] = np.nan

    # Patrial test TODO - remove
    #df_with_gt = df_with_gt[df_with_gt['ticker'].isin(sorted(list(set(df_with_gt['ticker'].values)))[:10])]
    #df_with_gt = df_with_gt[df_with_gt['ticker'].isin(['ENPH','SMCI', 'AXON' , 	'BLDR'])]


    for ticker, sdf in  df_with_gt.groupby(['ticker','predictor']):
        sdf = sdf.sort_values(by='date')
        for t in sorted(prediction_times):
            sdf[f"pred_gt{t}"] = np.hstack([sdf.close.values[t:], np.ones(t, ) * sdf.close.values[-1]])
            #sdf[f"pred_err{t}"] =np.abs(sdf[f"pred{t}" ].values - sdf[f"pred_gt{t}"].values) / sdf['close'].values[0] # error normalized to the first price
            sdf[f"pred_err{t}"] = np.abs(sdf[f"pred{t}"].values - sdf[f"pred_gt{t}"].values) / sdf['close'].values  # error as percentage of the price
            df_with_gt.loc[sdf.index, [f"pred_gt{t}"]] = sdf[f"pred_gt{t}"]
            df_with_gt.loc[sdf.index, [f"pred_err{t}"]] = sdf[f"pred_err{t}"]



    report = HtmlReport()
    for ticker, sdf in  df_with_gt.groupby('ticker'):
        print(ticker)
        for t in sorted(prediction_times):

            fig = plt.figure(figsize=(20, 15))
            for i, (predictor, pdf) in enumerate(sdf.groupby('predictor')):
                if i == 0:
                    plt.plot(pdf.date.values,pdf.close.values ,label = f"price")
                    plt.plot(pdf.date.values,pdf[f"pred_gt{t}"].values, label=f"gt pred {t}")
                plt.plot(pdf.date.values,pdf[f"pred{t}" ].values,label = f" pred {t} {predictor}")
            plt.legend()
            report.add_figure(f"{ticker} pred {t}", fig)
            plt.close("all")


    for t in sorted(prediction_times):
        fig, ax = plt.subplots(figsize=(20, 15 ))
        sns.boxplot(data=df_with_gt.dropna(subset=[f"pred_err{t}"]), x='predictor', y=f"pred_err{t}", ax=ax)
        report.add_figure(f"pred_err{t}" ,fig )



    res = []
    res_per_stock = []
    for pred, sdf in  df_with_gt.groupby('predictor'):
        for t in sorted(prediction_times):
            perr = sdf.dropna(subset=[f"pred_err{t}"])

            res.append({'predictor':pred, 't_pred' : t , 'rms' : np.sqrt(np.mean(perr[f"pred_err{t}"].values**2)),
                        'median': np.median(perr[f"pred_err{t}"].values)})
            for ticker, sperr in  perr.groupby('ticker'):
                res_per_stock.append({'predictor':pred,'ticker' : ticker, 't_pred' : t , 'rms' : np.sqrt(np.mean(sperr[f"pred_err{t}"].values**2)),
                            'median': np.median(sperr[f"pred_err{t}"].values)})

    #print(pd.DataFrame(res_per_stock))
    print(pd.DataFrame(res))
    report.add_df('error per ticker', pd.DataFrame(res_per_stock))
    report.add_df('results (error as percentage of price)', pd.DataFrame(res))
    report.to_file(os.path.join(outputdir , f'prediction_metric_report_{dbname}.html'))

def run_metrics(predictors):
    # Prevent sleep
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    params = get_config()


    dbname = params['db']
    prediction_times = [5]
    results_dir = f"C:/Users/dadab/projects/algotrading/results/inference"
    outputdir = f"C:/Users/dadab/projects/algotrading/results/eval"
    run_prediction_metrics(dbname, predictors, results_dir, outputdir, prediction_times)


if __name__ == "__main__":
    run_metrics(predictors= ['lstm1',  'simp_tf', 'tft', 'rls'])
    #run_metrics(predictors=['tft'])