import json
import os
import sys
from typing import Dict
import pickle
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
# import lightning as L
# import lightning.pytorch.callbacks as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting.metrics import QuantileLoss , SMAPE

from loaders.dataloaders import get_loaders
from models.get_models import get_model
from config.config import get_config

def run_training(datadir : str ,outdir: str ,params : Dict ):

    # Prevent sleep
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    max_epochs = params['max_epoch']
    checkpoint_to_load = params['checkpoint_to_load']
    checkpoints_path = os.path.join(outdir, 'checkpoints')
    log_path = os.path.join(outdir, 'logs')
    os.makedirs(checkpoints_path ,exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor="val_loss",   #"val_loss", # metric to track
        mode="min",               # "min" if lower is better (e.g., loss)
        save_top_k=1,             # save only the best model
        filename="best-checkpoint",  # name of the file
        save_last=True,
        verbose=True
    )

    torch.set_float32_matmul_precision('medium')

    print('get loaders')
    train_loader, val_loader= get_loaders(datadir, max_prediction_length = params['pred_len'] , max_encoder_length = params['max_encoder_length']
                                          , step=params['train_data_step']  ,  loader_type = params['model_type'], batch_size=params['batch_size'],
                                          features =params['features'])

    print('get model ')
    model = get_model(params['model_type'], params,train_loader.dataset , checkpoint_to_load=checkpoint_to_load )

    print('start training')
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1 ,     callbacks=[checkpoint_callback],
                         default_root_dir=log_path,
                         gradient_clip_val=1.0,
                         num_sanity_val_steps=0,
                         fast_dev_run=False,
                         )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":

    params = get_config()
    datadir = f'C:/Users/dadab/projects/algotrading/data/training/{ params['db']}/'
    outdir = f"C:/Users/dadab/projects/algotrading/training/{params['run_name']}_{params['model_type']}"

    run_training(datadir, outdir ,  params=params)
