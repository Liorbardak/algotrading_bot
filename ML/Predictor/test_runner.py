import os
import argparse
from config.config import get_config
from run_inference import run_inference
from run_training import run_training
from ML.Predictor.metrics.prediction_metrics import run_prediction_metrics
from ML.Predictor.models.rls_predictor import run_simple_predictors
def run_serial_tests(dataset : str = "" ,  predictors = ['simp_tf' , 'lstm1', 'lstm2']):
    # run list of trainings/inference/metric
    basedir = 'C:/Users/dadab/projects/algotrading'
    params = get_config()
    if dataset != "":
        params['db'] = dataset


    datadir = f'{basedir}/data/training/{params['db']}/'
    # for predictor in predictors:
    #     params['model_type'] = predictor
    #     outdir = f"{basedir}/training/{params['db']}_{params['model_type']}"
    #     inference_outdir = f"{basedir}/results/inference/{params['db']}_{params['model_type']}"
    #
    #     params['checkpoint_to_load'] = ""
    #     run_training(datadir, outdir, params)
    #
    #     params['checkpoint_to_load'] = os.path.join(outdir, "checkpoints", "best-checkpoint.ckpt")
    #     run_inference(datadir, inference_outdir, params)

    # Run rls & no prediction
    run_simple_predictors(params['db'])



    prediction_times = [params['pred_len']]  # test only for the maximal prediction time
    results_dir = f"C:/Users/dadab/projects/algotrading/results/inference"
    metric_output = f"C:/Users/dadab/projects/algotrading/results/eval"
    run_prediction_metrics(params['db'], predictors + ['rls','no_prediction'], results_dir, metric_output, prediction_times)





def main():
    basedir = 'C:/Users/dadab/projects/algotrading'

    parser = argparse.ArgumentParser(description='Run algo trading sim')
    parser.add_argument('-db', type=str,default="", help='db name  ')
    parser.add_argument('-m', type=str,default="",  help='model type')
    parser.add_argument('-cp', type=str, default="", help='checkpoint to load ')
    parser.add_argument('-t', type=str, default="all", help='running type -  all, training , inference')
    args = parser.parse_args()

    params = get_config()
    if args.db != "":
        params['db'] =args.db
    if args.m != "":
        params['model_type'] = args.m


    datadir = f'{basedir}/data/training/{params['db']}/'
    outdir = f"{basedir}/training/{params['db']}_{params['model_type']}"
    inference_outdir = f"{basedir}/results/inference/{params['db']}_{params['model_type']}"
    checkpoint_to_load =  args.cp


    if (args.t == "all") or (args.t == "training") :
        run_training(datadir, outdir, params)
    if (args.t == "all") or (args.t == "inference") :
        if checkpoint_to_load == "":
            params['checkpoint_to_load'] = os.path.join(outdir, "checkpoints", "best-checkpoint.ckpt")
        run_inference(datadir, inference_outdir, params)


if __name__ == "__main__":
    run_serial_tests("snp_v5")
    #main()