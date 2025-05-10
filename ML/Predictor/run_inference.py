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
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from loaders.dataloaders import get_loaders
from models.get_models import get_model




def run_inference(datadir, outputdir , checkpoint_to_load, params ,display = False , dbname = 'test_stocks.csv'):
    '''
    Run inference and save results
    :param datadir:
    :param outputdir:
    :param checkpoint_to_load:
    :param params:
    :return:
    '''

    if  params['model_type'] == 'tft':
        run_inference_tft(datadir, outputdir, checkpoint_to_load, params, display = display , dbname = dbname)
    else:
        run_inference_simple(datadir, outputdir , checkpoint_to_load, params, display = display ,  dbname = dbname )

    # Combine with the original data

    # Get the original stock prices
    df = pd.read_csv(os.path.join(db_path,'test_df_orig.csv'))
    # Get the predicted stock prices , just calculated here
    pred_df = pd.read_csv(os.path.join(outdir,'predictions.csv'))

    # merge
    merged_df = pd.merge(df, pred_df, on=['ticker','date'], how='outer')
    merged_df.to_csv(os.path.join(outdir,'ticker_data.csv'))



def run_inference_simple(datadir, outputdir , checkpoint_to_load, params ,display = False , dbname = 'train_stocks.csv'):
    '''
    Run inference and save results
    :param datadir:
    :param outputdir:
    :param checkpoint_to_load:
    :param params:
    :return:
    '''

    batch_size = 32

    os.makedirs(outputdir, exist_ok=True)
    # load normalization factors
    normalization_factor = pickle.load(open(os.path.join(datadir,dbname.split('.')[0] + '_norm_factors.pkl'),'rb'))

    # Load the model

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
            ids_to_show = [np.argmax(np.mean((output_cpu - output_cpu_gt) ** 2, axis=1))]  # worth sample
            # ids_to_show = range(output_cpu.shape[0])

            for idx in range(output_cpu.shape[0]):

                data = inference_loader.dataset.get_meta(idx + bidx)

                normfact = normalization_factor[data['stock'] + 'normFact']

                r = {'ticker': data['stock'], 'date': data['date']}
                for i, pred in enumerate(output_cpu[idx]):
                    r['pred' + str(i)] = pred / normfact
                rows.append(r)

                if(display) & (idx in ids_to_show) :
                    print(loss)

                    plt.figure()
                    plt.plot(input_cpu[idx,:,3],label='input')

                    #plt.plot(np.arange(len(input_cpu[idx,:,3]),len(input_cpu[idx,:,3]) + len(output_cpu_gt[idx])), output_cpu_gt[idx])
                    plt.plot(np.arange(len(input_cpu[idx, :, 3]), len(input_cpu[idx, :, 3]) + len(output_cpu_gt[idx])),
                             output_cpu_gt[idx],label='gt')
                    plt.plot(np.arange(len(input_cpu[idx, :, 3]), len(input_cpu[idx, :, 3]) + len(output_cpu_gt[idx])),
                             output_cpu[idx] ,label='out')
                    plt.title(data['stock'])
                    plt.legend()
                    plt.show()
            bidx += batch_size

    # Save predictions
    pd.DataFrame(rows).to_csv(os.path.join(outputdir, 'predictions.csv'))


def run_inference_tft(datadir, outputdir , checkpoint_to_load, params, display=False , dbname = 'train_stocks.csv' ):
    '''
    Run inference and save results with tft model
    :param datadir:
    :param outputdir:
    :param checkpoint_to_load:
    :param params:
    :return:
    '''
    torch.set_float32_matmul_precision('medium')

    batch_size = 32

    os.makedirs(outputdir, exist_ok=True)

    normalization_factor = pickle.load(open(os.path.join(datadir,dbname.split('.')[0] + '_norm_factors.pkl'),'rb'))
    stock_data = pd.read_csv(os.path.join(datadir,dbname))
    # load model dataset params -
    #dataset_parameters = pickle.load(open(os.path.join(os.path.dirname(checkpoint_to_load), 'model_dataset_params.pkl'), 'rb'))

    # ID to name
    id_to_name = dict()
    for n, df in stock_data.groupby('ticker'):
        id_to_name[df['ticker_id'].values[0]] = n

    inference_loader  = get_loader(datadir, dbname, max_prediction_length = params['pred_len'] , max_encoder_length = params['max_encoder_length']
                                   , batch_size=batch_size , shuffle=False, get_meta = True, loader_type = params['model_type']  )

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(params['model_type'], params,inference_loader.dataset , checkpoint_to_load )

    # Predict
    prediction_items  = model.predict(inference_loader, return_x=True)
    predictions = prediction_items[0].cpu().numpy()
    x = prediction_items[1]
    group_ids = x['groups'].cpu().numpy().flatten()
    decoder_time_idx = x['decoder_time_idx'].cpu().numpy()

    # Translate prediction to data frame
    rows = []
    for prediction,group_id, time_idx in  zip(predictions,group_ids,decoder_time_idx):
        ticker = id_to_name[group_id]
        date = stock_data[(stock_data.ticker_id == group_id) & (stock_data.time_idx == time_idx[0]-1)].date.values[0]

        normfact = normalization_factor[ticker + 'normFact']
        r = {'ticker':ticker , 'date': date }
        for i, pred in enumerate(prediction):
            r['pred' + str(i)] = pred / normfact
        rows.append(r)
        if(display):
            if time_idx[0] % 20 == 0:
                # Get the data before the prediction time (for display )
                prev_idx =np.arange(time_idx[0]-20,time_idx[0])
                prev_v = []
                for ix in prev_idx:
                    prev_v.append(stock_data[(stock_data.ticker_id == group_id) & (stock_data.time_idx == ix)]['close'])

                plt.figure()
                plt.plot(time_idx,  prediction ,label='prediction')
                plt.plot(prev_idx, prev_v, label='before')
                plt.title(f' {ticker}  {time_idx[0]}')
                plt.show()


    pd.DataFrame(rows).to_csv(os.path.join(outputdir,'predictions.csv'))

def main():
    params = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'training_config.json')))

    dbtrain_name =  'snp_v0'
    db_path = f'C:/Users/dadab/projects/algotrading/data/training/{dbtrain_name}/'   #path of
    checkpoint_path = f"C:/Users/dadab/projects/algotrading/training/{dbtrain_name}_{params['model_type']}"
    outdir = f"C:/Users/dadab/projects/algotrading/results/{dbtrain_name}_{params['model_type']}"
    os.makedirs(outdir, exist_ok=True)

    checkpoint_to_load = os.path.join(checkpoint_path,"checkpoints", "best-checkpoint.ckpt") # Note - dont forget to change the checkpoint name
    run_inference(db_path, outdir , checkpoint_to_load, params , display = False )


if __name__ == "__main__":

    params = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'training_config.json')))
    dbtrain_name = 'snp_v0'
    db_path = f'C:/Users/dadab/projects/algotrading/data/training/{dbtrain_name}/'   #path of
    checkpoint_path = f"C:/Users/dadab/projects/algotrading/training/{dbtrain_name}_{params['model_type']}"
    outdir = f"C:/Users/dadab/projects/algotrading/results/{dbtrain_name}_{params['model_type']}"

    main()
