import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pytorch_forecasting.metrics import QuantileLoss , SMAPE

from tft_predictor import  TFTLightningWrapper
from transformer_predictor import  LitStockPredictor ,TransformerPredictorModel
from simple_predictor import LinearPredictorModel

def get_model(model_type : str, parameters : dict , dataset = None , checkpoint_to_load : str = None):
    if model_type == 'simp_tf':
        if checkpoint_to_load is None:
            model = LitStockPredictor(model=TransformerPredictorModel(pred_len=parameters['pred_len']) ,  params  = {'lr' : parameters['lr'] ,'loss': nn.MSELoss()})
        else:
            model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load,
                                                           model=TransformerPredictorModel(pred_len=parameters['pred_len']),
                                                           params  = {'lr' : parameters['lr'] ,'loss': nn.MSELoss()})
    elif model_type == 'regressor':
        if checkpoint_to_load is None:
            model = LitStockPredictor(model=LinearPredictorModel(pred_len=parameters['pred_len']) ,  params  = {'lr' : parameters['lr'] ,'loss': nn.MSELoss()})
        else:
            model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load,
                                                           model=LinearPredictorModel(pred_len=parameters['pred_len']),
                                                           params  = {'lr' : parameters['lr'] ,'loss': nn.MSELoss()})
    elif model_type == 'tft':
        if checkpoint_to_load is None:
            model = TFTLightningWrapper(
                training_dataset=dataset,
                learning_rate=parameters['lr'],
                hidden_size=16,
                attention_head_size=1,
                dropout=0.1,
                hidden_continuous_size=8,
                output_size=1,
                loss=SMAPE(reduction="mean"),
                log_interval=10,
                reduce_on_plateau_patience=4,
            )
        else:
            model = TFTLightningWrapper.load_from_checkpoint(
                checkpoint_to_load,
                training_dataset=dataset,
            )
    else:
        raise ValueError('model_type unrecognized')
    return model
