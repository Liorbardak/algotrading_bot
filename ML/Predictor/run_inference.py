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
from lightingwraper import  LitStockPredictor
#from models.transformer_predictor import TransformerPredictorModel


checkpoint_path = "C:/Users/dadab/projects/algotrading/training/test_good/checkpoints"
#datadir = 'C:/Users/dadab/projects/algotrading/data/training/dbsmall'
datadir = 'C:/Users/dadab/projects/algotrading/data/training/allbad'

outputdir = 'C:/Users/dadab/projects/algotrading/data/training/predictions'


#dbname = 'train_stocks.csv'
dbname = 'val_stocks.csv'


os.makedirs(outputdir, exist_ok=True)

# Load model
checkpoint_to_load = os.path.join(checkpoint_path ,"best-checkpoint.ckpt")
# Get normalization factors
normalization_factor = pickle.load(open(os.path.join(datadir,'norm_factors.pkl'),'rb'))

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load)
model.to(device)
model.eval()
batch_size = 64
max_prediction_length = 15
inference_loader  = get_loader(datadir, dbname, max_prediction_length = max_prediction_length , max_encoder_length = 60
                               , batch_size=batch_size , shuffle=False, get_meta = True )


predictions = []
with torch.no_grad():
    bidx = 0
    for x, y in inference_loader:
        input_cpu = x.numpy()
        output_cpu_gt =  y.numpy()
        x = x.to(device)
        output = model(x)
        output_cpu = output.cpu().numpy()
        loss = np.mean((output_cpu-output_cpu_gt)**2)
        #print(loss)
        if(loss > 1):
            print(loss)

            idx = np.argmax(np.mean((output_cpu - output_cpu_gt) ** 2, axis=1))
            print(dat['stock'] , idx)

            dat = inference_loader.dataset.get_meta(idx + bidx)
            plt.figure()
            plt.plot(input_cpu[idx,:,3],label='input')

            #plt.plot(np.arange(len(input_cpu[idx,:,3]),len(input_cpu[idx,:,3]) + len(output_cpu_gt[idx])), output_cpu_gt[idx])
            plt.plot(np.arange(len(input_cpu[idx, :, 3]), len(input_cpu[idx, :, 3]) + len(output_cpu_gt[idx])),
                     output_cpu_gt[idx],label='gt')
            plt.plot(np.arange(len(input_cpu[idx, :, 3]), len(input_cpu[idx, :, 3]) + len(output_cpu_gt[idx])),
                     output_cpu[idx] ,label='out')
            plt.title(dat['stock'])
            plt.legend()
            plt.show()
      # sort out , renormalize
        for idx in np.arange(output.shape[0]):
            # get the output meta data
            dat = inference_loader.dataset.get_meta(idx+bidx)
            # get the normalization factor
            normfact = normalization_factor[dat['stock']+'normFact']
            for i in np.arange(output_cpu.shape[1]):
                dat['pred' + str(i)] = output_cpu[idx, i] / normfact
            predictions.append(dat)
        bidx = bidx + output.shape[0]
    plt.show()
df = pd.DataFrame(predictions)
df.to_csv(os.path.join(outputdir,'predictions.csv'))