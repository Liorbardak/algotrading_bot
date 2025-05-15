import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pytorch_forecasting.metrics import QuantileLoss , SMAPE

from tft_predictor import  TFTLightningWrapper
from transformer_predictor import  LitStockPredictor , TimeSeriesTransformer
from simple_predictor import LinearPredictorModel
from lstm_predictor import Seq2SeqLSTM, Seq2SeqLSTM2

def get_model(model_type : str, parameters : dict , dataset = None , checkpoint_to_load : str = None):
    if model_type == 'simp_tf':
        if checkpoint_to_load == "":
            model = LitStockPredictor(model=TimeSeriesTransformer(input_size = parameters['input_len'] , pred_len=parameters['pred_len']) ,  params  = {'lr' : parameters['lr'] ,'loss': nn.MSELoss()})
        else:
            model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load,
                                                           model=TimeSeriesTransformer(input_size = parameters['input_len'] ,pred_len=parameters['pred_len']),
                                                           params  = {'lr' : parameters['lr'] ,'loss': nn.MSELoss()})

    elif model_type == 'lstm1':
        if checkpoint_to_load == "":
            model = LitStockPredictor(model=Seq2SeqLSTM( input_size=parameters['input_len'], hidden_size=32, output_len=parameters['pred_len']) ,  params  = {'lr' : parameters['lr'] ,'loss': nn.L1Loss()})
        else:
            model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load,
                                                           model=Seq2SeqLSTM( input_size=parameters['input_len'], hidden_size=32, output_len=parameters['pred_len']),
                                                           params  = {'lr' : parameters['lr'] ,'loss': nn.L1Loss()})

    elif model_type == 'lstm2':
        if checkpoint_to_load == "":
            model = LitStockPredictor(model=Seq2SeqLSTM( input_size=parameters['input_len'], hidden_size=64, output_len=parameters['pred_len']) ,  params  = {'lr' : parameters['lr'] ,'loss': nn.L1Loss()})
        else:
            model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load,
                                                           model=Seq2SeqLSTM( input_size=parameters['input_len'], hidden_size=64, output_len=parameters['pred_len']),
                                                           params  = {'lr' : parameters['lr'] ,'loss': nn.L1Loss()})

    elif model_type == 'regressor':
        if checkpoint_to_load == "":
            model = LitStockPredictor(model=LinearPredictorModel(pred_len=parameters['pred_len']) ,  params  = {'lr' : parameters['lr'] ,'loss': nn.L1Loss()})
        else:
            model = LitStockPredictor.load_from_checkpoint(checkpoint_to_load,
                                                           model=LinearPredictorModel(pred_len=parameters['pred_len']),
                                                           params  = {'lr' : parameters['lr'] ,'loss': nn.L1Loss()})
    elif model_type == 'tft':
        if checkpoint_to_load == "":
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
