import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from lightning.pytorch import LightningModule


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=config['trainer_params']['devices'] != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy="ddp",
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)