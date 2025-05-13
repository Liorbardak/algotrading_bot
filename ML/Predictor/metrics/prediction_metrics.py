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

def run_prediction_metrics(dbname: str, predictors: str, results_dir : str, outputdir : str , prediction_times : List , display : bool = False ):

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


    for ticker, sdf in  df_with_gt.groupby(['ticker','predictor']):
        sdf = sdf.sort_values(by='date')
        for t in sorted(prediction_times):
            sdf[f"pred_gt{t}"] = np.hstack([sdf.close.values[t:], np.ones(t, ) * sdf.close.values[-1]])
            sdf[f"pred_err{t}"] =np.abs(sdf[f"pred{t}" ].values - sdf[f"pred_gt{t}"].values) / sdf['close'].values[0] # error normalized to the first price
            df_with_gt.loc[sdf.index, [f"pred_gt{t}"]] = sdf[f"pred_gt{t}"]
            df_with_gt.loc[sdf.index, [f"pred_err{t}"]] = sdf[f"pred_err{t}"]


    #df_with_gt = df_with_gt[df_with_gt['ticker'] == df_with_gt['ticker'].values[0]]

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
    for pred, sdf in  df_with_gt.groupby('predictor'):
        for t in sorted(prediction_times):
            perr = sdf.dropna(subset=[f"pred_err{t}"])
            print(perr[f"pred_err{t}"].values.mean() )
            res.append({'predictor':pred, 't_pred' : t , 'rms' : np.sqrt(np.mean(perr[f"pred_err{t}"].values**2)),
                        'median': np.median(perr[f"pred_err{t}"].values)})
    print(pd.DataFrame(res))
    report.add_df('results (error normalized to first price)' ,pd.DataFrame(res) )
    report.to_file(os.path.join(outputdir , f'prediction_metric_report_{dbname}.html'))



if __name__ == "__main__":
    params = get_config()
    params = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'training_config.json')))

    dbname = params['db']
    predictors = ['lstm1','lstm2','simp_tf' , 'tft', 'rls']
    predictors = ['tft']
    prediction_times = [1,2,5]
    results_dir = f"C:/Users/dadab/projects/algotrading/results/inference"
    outputdir = f"C:/Users/dadab/projects/algotrading/results/eval"
    run_prediction_metrics(dbname, predictors, results_dir,outputdir, prediction_times)
