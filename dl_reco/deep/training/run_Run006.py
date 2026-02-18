import data_gen
import hyperparameters
import models


import os
import sys
import argparse
import shutil
import numpy as np
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
#from lightning.pytorch import seed_everything

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging

"""
Calls hyperparameters, models and data_gen. Trains the model on the specidied training data and saves the trained model. Can be used in the future to evaluate the model and plot the results.
"""

#---------create file structure
parser = argparse.ArgumentParser()
parser.add_argument("--Run_number", type=str, default='RunXXX', help="Name a new Run to be saved")
args = parser.parse_args()

saved_dir = 'results/' + args.Run_number
saved_model_dir = 'results/' + args.Run_number + '/model'
saved_plots_dir = 'results/' + args.Run_number + '/plots'

if saved_dir == 'results/RunXXX': 
    if not os.path.exists(saved_plots_dir):
        os.makedirs(saved_plots_dir)
    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)
else:
    assert(not os.path.exists(saved_dir))
    os.makedirs(saved_plots_dir)
    os.mkdir(saved_model_dir)


src_hyper = 'hyperparameters.py'
dst_hyper = 'results/' + args.Run_number + '/hyperparameters_' + args.Run_number + '.py'
shutil.copy(src_hyper, dst_hyper)

src_models= 'models.py'
dst_models = 'results/' + args.Run_number + '/models_' + args.Run_number + '.py'
shutil.copy(src_models, dst_models)

src_run = 'run.py'
dst_run = 'results/' + args.Run_number + '/run_' + args.Run_number + '.py'
shutil.copy(src_run, dst_run)

src_data = 'data_gen.py'
dst_data = 'results/' + args.Run_number + '/data_' + args.Run_number + '.py'
shutil.copy(src_data, dst_data)



#---------load the data

list_of_file_ids_train = np.arange(hyperparameters.n_files_train, dtype=int)
list_of_file_ids_val = np.arange(hyperparameters.n_files_val, dtype=int)


train = data_gen.Prepare_Dataset_direction(file_ids=list_of_file_ids_train, points = hyperparameters.n_events_per_file, data_type = '_training', worker_num=hyperparameters.worker_num)
val = data_gen.Prepare_Dataset_direction(file_ids=list_of_file_ids_val, points = hyperparameters.n_events_per_file, data_type = '_validating', worker_num=hyperparameters.worker_num)


train_loader = DataLoader(train, batch_size=hyperparameters.batchSize, shuffle=False, num_workers=hyperparameters.worker_num)
val_loader = DataLoader(val, batch_size=hyperparameters.batchSize, shuffle=False, num_workers=hyperparameters.worker_num)

#---------load the model

if hyperparameters.model_name == "Nils":
    model = models.Nils(loss_function = hyperparameters.loss_function)

elif hyperparameters.model_name == "GW":
    model = models.GW(loss_function = hyperparameters.loss_function)

elif hyperparameters.model_name == "Split":
    model = models.Split(loss_function = hyperparameters.loss_function)

#model.bfloat16()#.float() #bfloat
#for param in model.parameters():
#    param.data = param.data.to(torch.bfloat16)
model.float()
#model.double()




model.pdf_energy_vertex.double()
model.pdf_direction.double()
#model.pdf_vertex.double()

#seed_everything(hyperparameters.seed)
#torch.use_deterministic_algorithms(True, warn_only=True)



#---------define callbacks
#mc = ModelCheckpoint(dirpath=saved_model_dir, filename= '{epoch}_model_checkpoint', 
#    monitor='val_loss', verbose=1, save_top_k=3)

#mc = ModelCheckpoint(dirpath=saved_model_dir, filename= 'model_checkpoint', 
#    monitor='val_total_loss', verbose=1, save_top_k=3)

mc = ModelCheckpoint(dirpath=saved_model_dir, filename= '{epoch}_model_checkpoint', 
    monitor='val_total_loss', verbose=1, every_n_epochs = 1, save_top_k = -1)#, save_top_k=3)

es = EarlyStopping("val_total_loss", patience=hyperparameters.es_patience, min_delta=hyperparameters.es_min_delta, verbose=1)

swa = StochasticWeightAveraging(swa_lrs=1e-2)

lr_monitor = LearningRateMonitor(logging_interval='step')

callbacks = [es, mc, swa, lr_monitor]
#callbacks = [lr_monitor]
tb_logger = TensorBoardLogger(saved_model_dir, name="tb_logger", version='version1_')

trainer = pl.Trainer(
    devices=1, 
    #auto_select_gpus=True,
    callbacks = callbacks, 
    max_epochs = hyperparameters.epochs,
    log_every_n_steps=1,
    logger = tb_logger,
    precision = 32#,
    #deterministic=True
    )

cont_counter = 0
max_counter = 100

best_checkpoint_path = None

while cont_counter < max_counter:
    #trainer.fit(model, train_loader, val_loader, ckpt_path = best_checkpoint_path)

    try:

        #---------training
        trainer = pl.Trainer(
            devices=1, 
            #auto_select_gpus=True,
            #gradient_clip_val=0.5,
            callbacks = callbacks, 
            max_epochs = hyperparameters.epochs,
            log_every_n_steps=1,
            logger = tb_logger,
            precision = 32#,
            #deterministic=True
            )
        trainer.fit(model, train_loader, val_loader, ckpt_path = best_checkpoint_path)
        break
    
    except Exception as error:
        cont_counter += 1
        print('error: ', error)
        print('Unexpected error, starting continuation = ', cont_counter)

        best_checkpoint_path = saved_model_dir + '/version1_model_checkpoint.ckpt'

        if not os.path.isfile(best_checkpoint_path):
            
            print('starting from the beginning, cont= ', cont_counter)

            best_checkpoint_path = None

        trainer.fit(model, train_loader, val_loader, ckpt_path = best_checkpoint_path)
    
#---------evaluate model

#---------save model  
torch.save(model.state_dict(), saved_model_dir + '/model_' + args.Run_number + '.pt')

