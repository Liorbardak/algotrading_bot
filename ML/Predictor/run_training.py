import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from loaders.dataloaders import get_loaders
from lightingwraper import  LitStockPredictor


def run_training(datadir : str ,outdir: str , max_epochs=2 ,ckpt_path=None):
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

    max_prediction_length = 15
    data_step = 10 # take every step sample
    torch.set_float32_matmul_precision('medium')
    print('get loaders')
    train_loader, val_loader= get_loaders(datadir, max_prediction_length = max_prediction_length , max_encoder_length = 60
                                          , step=data_step  )
    print('get predictor ')
    model = LitStockPredictor()
    #model = LitStockPredictor(model=TransformerPredictorModel(pred_len=max_prediction_length))
    print('start training')
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1 ,     callbacks=[checkpoint_callback],
                         default_root_dir=log_path,
                         gradient_clip_val=1.0,
                         num_sanity_val_steps=0,
                         fast_dev_run=False,
                         )
    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    datadir = 'C:/Users/dadab/projects/algotrading/data/training/good2'
    outdir = "C:/Users/dadab/projects/algotrading/training/playground"
    run_training(datadir, outdir , max_epochs=300)
    # train_dataloader, val_dataloader, training = get_loaders(datadir, max_prediction_length = 20 , max_encoder_length = 60 ,     batch_size = 64 )
    # run_training(train_dataloader, val_dataloader, training)