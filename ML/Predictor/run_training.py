import json
import os
import sys
from typing import Dict
import pandas as pd
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting.metrics import QuantileLoss , SMAPE

from loaders.dataloaders import get_loaders
from models.get_models import get_model

def run_training(datadir : str ,outdir: str ,params : Dict ,max_epochs=2 ):
    checkpoints_path = os.path.join(outdir, 'checkpoints')
    log_path = os.path.join(outdir, 'logs')
    os.makedirs(checkpoints_path ,exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor="val_loss",       # metric to track
        mode="min",               # "min" if lower is better (e.g., loss)
        save_top_k=1,             # save only the best model
        filename="best-checkpoint",  # name of the file
        save_last=True,
        verbose=True
    )

    data_step = 10 # take every step sample
    torch.set_float32_matmul_precision('medium')

    print('get loaders')
    train_loader, val_loader= get_loaders(datadir, max_prediction_length = params['pred_len'] , max_encoder_length = params['max_encoder_length']
                                          , step=data_step ,  loader_type = params['model_type'] )

    print('get model ')
    model = get_model(params['model_type'], params,train_loader.dataset )

    print('start training')
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1 ,     callbacks=[checkpoint_callback],
                         default_root_dir=log_path,
                         gradient_clip_val=1.0,
                         num_sanity_val_steps=0,
                         fast_dev_run=False,
                         )
    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    params = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'training_config.json')))
    datadir = 'C:/Users/dadab/projects/algotrading/data/training/dbsmall'
    outdir = "C:/Users/dadab/projects/algotrading/training/dbsmall_" + params['model_type']


    run_training(datadir, outdir , max_epochs=50, params=params )
