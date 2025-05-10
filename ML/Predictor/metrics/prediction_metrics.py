
import os
import numpy as np
import pandas as pd
import pylab as plt
from typing import List
import seaborn as sns
from basic_code.utils.report_utils import HtmlReport

def run_prediction_metrics(dbname: str, predictors: str, results_dir : str, outputdir : str , prediction_times : List , display : bool = False ):
    df_with_gt = []
    for predictor in predictors:
        report = HtmlReport()

        df = pd.read_csv(os.path.join(results_dir, f"{dbname}_{predictor}", "ticker_data.csv" ))

        for ticker, sdf in  df.groupby('ticker'):
            sdf = sdf.sort_values(by='date')
            for t in sorted(prediction_times):
                sdf['predictor'] = predictor
                sdf[f"pred_gt{t}"] = np.hstack([sdf.close.values[t:], np.ones(t,)*sdf.close.values[-1]])

                #  prediction error (normalized to the first  close value
                sdf[f"pred_err{t}"] = np.abs(sdf[f"pred{t}" ].values - sdf[f"pred_gt{t}"].values) / sdf['close'].values[0]


                fig = plt.figure(figsize = (20,15))
                plt.plot(sdf.close.values ,label = f"price")
                plt.plot(sdf[f"pred{t}" ].values,label = f" pred {t}")
                plt.plot(sdf[f"pred_gt{t}"].values ,label = f"gt pred {t}")
                plt.legend()
                plt.title(ticker)
                report.add_figure(f"{ticker} {predictor} pred {t}", fig)
                plt.close("all")
            df_with_gt.append(sdf)
    df_with_gt = pd.concat(df_with_gt)

    for t in sorted(prediction_times):
        fig, ax = plt.subplots(figsize=(20, 15 ))
        sns.boxplot(data=df_with_gt.dropna(subset=[f"pred_err{t}"]), x='predictor', y=f"pred_err{t}", ax=ax)
        report.add_figure(f"pred_err{t}" ,fig )
    report.to_file(os.path.join(outputdir , 'prediction_metric_report.html'))












if __name__ == "__main__":
    dbname = 'snp_v0'
    predictors = ['simp_tf']
    prediction_times = [5]
    results_dir = f"C:/Users/dadab/projects/algotrading/results"
    outputdir = f"C:/Users/dadab/projects/algotrading/results/eval"
    run_prediction_metrics(dbname, predictors, results_dir,outputdir, prediction_times)
