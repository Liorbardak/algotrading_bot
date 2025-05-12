import os
import argparse
from config.config import get_config
from run_inference import run_inference
from run_training import run_training
def run_serial_tests():
    # run list of trainings
    basedir = 'C:/Users/dadab/projects/algotrading'
    params = get_config()
    #for model in ["simp_tf" , 'tft' ,'lstm1' ,'lstm2']:
    for model in ['tft']:
        params['model_type'] = model
        datadir = f'{basedir}/data/training/{params['db']}/'
        outdir = f"{basedir}/training/{params['db']}_{params['model_type']}"
        inference_outdir = f"{basedir}/results/inference/{params['db']}_{params['model_type']}"

        run_training(datadir, outdir, params)
        params['checkpoint_to_load'] = os.path.join(outdir, "checkpoints", "best-checkpoint.ckpt")

        run_inference(datadir, inference_outdir, params)

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
    if  checkpoint_to_load == "":
        params['checkpoint_to_load'] = os.path.join(outdir, "checkpoints", "best-checkpoint.ckpt")

    if (args.t == "all") or (args.t == "training") :
        run_training(datadir, outdir, params)
    if (args.t == "all") or (args.t == "inference") :

        run_inference(datadir, inference_outdir, params)


if __name__ == "__main__":
    run_serial_tests()
    #main()