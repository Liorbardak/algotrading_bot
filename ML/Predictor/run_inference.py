import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import pylab as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import torch
#import pandas as pd
#from models.transformer_predictor import TransformerPredictorModel
from loaders.dataloaders import get_loader
from models.transformer_predictor import  LitStockPredictor

from loaders.dataloaders import get_loaders
from models.get_models import get_model



dbname = 'train_stocks.csv'

def run_inference(datadir, outputdir , checkpoint_to_load, params):
    '''
    Run inference and save results
    :param datadir:
    :param outputdir:
    :param checkpoint_to_load:
    :param params:
    :return:
    '''

    if  params['model_type'] == 'tft':
        run_inference_tft(datadir, outputdir, checkpoint_to_load, params)
    else:
        run_inference_simple(datadir, outputdir , checkpoint_to_load, params)

def run_inference_simple(datadir, outputdir , checkpoint_to_load, params ):
    '''
    Run inference and save results
    :param datadir:
    :param outputdir:
    :param checkpoint_to_load:
    :param params:
    :return:
    '''
    os.makedirs(outputdir, exist_ok=True)
    # load normalization factors
    normalization_factor = pickle.load(open(os.path.join(datadir,'norm_factors.pkl'),'rb'))

    # Load the model

    batch_size = 32
    inference_loader  = get_loader(datadir, dbname, max_prediction_length = params['pred_len'] , max_encoder_length = params['max_encoder_length']
                                   , batch_size=batch_size , shuffle=False, get_meta = True , loader_type = params['model_type'] )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(params['model_type'], params,inference_loader.dataset , checkpoint_to_load )
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        bidx = 0
        for x, y in inference_loader:
            input_cpu = x.numpy()
            output_cpu_gt =  y.numpy()
            x = x.to(device)
            output = model(x)
            output_cpu = output.cpu().numpy()
            loss = np.mean((output_cpu-output_cpu_gt)**2)

            # Store predictions
            for idx in range(output_cpu.shape[0]):

                data = inference_loader.dataset.get_meta(idx + bidx)

                normfact = normalization_factor[data['stock'] + 'normFact']

                r = {'name': data['stock'], 'Date': data['date']}
                for i, pred in enumerate(output_cpu[idx]):
                    r['pred' + str(i)] = pred / normfact
                rows.append(r)


                # print(loss)
                # if(loss > 0):
                #     print(loss)
                #
                #     idx = np.argmax(np.mean((output_cpu - output_cpu_gt) ** 2, axis=1))
                #
                #
                #
                #
                #     plt.figure()
                #     plt.plot(input_cpu[idx,:,3],label='input')
                #
                #     #plt.plot(np.arange(len(input_cpu[idx,:,3]),len(input_cpu[idx,:,3]) + len(output_cpu_gt[idx])), output_cpu_gt[idx])
                #     plt.plot(np.arange(len(input_cpu[idx, :, 3]), len(input_cpu[idx, :, 3]) + len(output_cpu_gt[idx])),
                #              output_cpu_gt[idx],label='gt')
                #     plt.plot(np.arange(len(input_cpu[idx, :, 3]), len(input_cpu[idx, :, 3]) + len(output_cpu_gt[idx])),
                #              output_cpu[idx] ,label='out')
                #     plt.title(data['stock'])
                #     plt.legend()
                #     plt.show()

    pd.DataFrame(rows).to_csv(os.path.join(outputdir,params['model_type'] + 'predictions.csv'))


def run_inference_tft(datadir, outputdir , checkpoint_to_load, params ):
    '''
    Run inference and save results with tft model
    :param datadir:
    :param outputdir:
    :param checkpoint_to_load:
    :param params:
    :return:
    '''

    os.makedirs(outputdir, exist_ok=True)

    normalization_factor = pickle.load(open(os.path.join(datadir,'norm_factors.pkl'),'rb'))
    stock_data = pd.read_csv(os.path.join(datadir,'train_stocks.csv'))

    # ID to name
    id_to_name = dict()
    for n, df in stock_data.groupby('stock_name'):
        id_to_name[df['stock_id'].values[0]] = n

    batch_size = 32
    inference_loader  = get_loader(datadir, dbname, max_prediction_length = params['pred_len'] , max_encoder_length = params['max_encoder_length']
                                   , batch_size=batch_size , shuffle=False, get_meta = True, loader_type = params['model_type']  )
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(params['model_type'], params,inference_loader.dataset , checkpoint_to_load )
    prediction_items  = model.predict(inference_loader, return_x=True)
    predictions = prediction_items[0].cpu().numpy()
    x = prediction_items[1]
    group_ids = x['groups'].cpu().numpy().flatten()
    decoder_time_idx = x['decoder_time_idx'].cpu().numpy()


    rows = []
    for prediction,group_id, time_idx in  zip(predictions,group_ids,decoder_time_idx):
        stock_name = id_to_name[group_id]
        date = stock_data[(stock_data.stock_id == group_id) & (stock_data.time_idx == time_idx[0]-1)].date.values[0]

        normfact = normalization_factor[stock_name + 'normFact']
        r = {'name':stock_name , 'Date': date }
        for i, pred in enumerate(prediction):
            r['pred' + str(i)] = pred / normfact
        rows.append(r)

        #
        # # Get the data before the prediction time
        # prev_idx =np.arange(time_idx[0]-20,time_idx[0])
        # prev_v = []
        # for ix in prev_idx:
        #     prev_v.append(stock_data[(stock_data.stock_id == group_id) & (stock_data.time_idx == ix)]['close'])
        #
        # plt.figure()
        # plt.plot(time_idx,  prediction ,label='prediction')
        # plt.plot(prev_idx, prev_v, label='before')
        # plt.title(f' {stock_name}  {time_idx[0]}')


    pd.DataFrame(rows).to_csv(os.path.join(outputdir,params['model_type'] + 'predictions.csv'))


if __name__ == "__main__":
    params = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'training_config.json')))
    outdir = "C:/Users/dadab/projects/algotrading/training/dbsmall_" + params['model_type']
    datadir = 'C:/Users/dadab/projects/algotrading/data/training/dbsmall'
    #checkpoint_path = os.path.join(outdir ,"checkpoints")
    checkpoint_to_load = os.path.join(outdir,"checkpoints" , "best-checkpoint.ckpt")

    run_inference(datadir, outdir , checkpoint_to_load, params )
