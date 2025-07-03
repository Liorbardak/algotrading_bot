
from config.config import get_config
from ML.Predictor.models.rls_predictor import run_simple_predictors
from ML.Predictor.metrics.prediction_metrics import run_prediction_metrics

def run_evaluation(predictors = None, add_simple_predictors : bool = True):
    # run  metric
    basedir = 'C:/Users/dadab/projects/algotrading'
    params = get_config()

    datadir = f'{basedir}/data/training/{params['db']}/'
    if predictors is None:
        predictors = [params['model_type']]

   # Run rls & no prediction
    if add_simple_predictors:
        predictors = predictors + ['rls', 'no_prediction']
        run_simple_predictors(params)



    prediction_times = [5]  #[params['pred_len']]  # test only for the maximal prediction time
    results_dir = f"C:/Users/dadab/projects/algotrading/results/inference"
    metric_output = f"C:/Users/dadab/projects/algotrading/results/eval"
    run_prediction_metrics(params, predictors, results_dir, metric_output, prediction_times)

if __name__ == "__main__":
    run_evaluation()