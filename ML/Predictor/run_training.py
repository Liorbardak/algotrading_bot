import os
import sys

#from torch._C.cpp import nn
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.transformer_predictor import TransformerPredictorModel
from loaders.dataloaders import get_loaders
from lightingwraper import  LitStockPredictor


def run_training(datadir : str ,outdir: str , max_epochs=2):
    checkpoints_path = os.path.join(outdir, 'checkpoints')
    log_path = os.path.join(outdir, 'logs')
    os.makedirs(checkpoints_path ,exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        monitor="val_loss",       # metric to track
        mode="min",               # "min" if lower is better (e.g., loss)
        save_top_k=3,             # save only the best model
        filename="best-checkpoint",  # name of the file
        verbose=True
    )


    train_loader, val_loader= get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 ,output_file = os.path.join(checkpoints_path,'scale.pkl')  )
    model = LitStockPredictor()
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1 ,     callbacks=[checkpoint_callback],
                         default_root_dir=log_path,
                         )
    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    datadir = 'C:/Users/dadab/projects/algotrading/data/training/dbmedium'
    outdir = "C:/Users/dadab/projects/algotrading/training/test"
    run_training(datadir, outdir , max_epochs=100)
    # train_dataloader, val_dataloader, training = get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 ,     batch_size = 64 )
    # run_training(train_dataloader, val_dataloader, training)