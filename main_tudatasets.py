# from model_gine_2 import Net
import math
from torch.cuda import is_available
from argparse import ArgumentParser
from datetime import datetime
from utils import get_run_path
from model_handler import ModelHandler
from data_handler import DataHandler, tu_dataset_type
from train_handler import get_trainer
from transforms import get_pre_transform, get_transform
import os
from model import Net
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from tqdm import tqdm

# from warnings import filterwarnings
# filterwarnings("ignore",category=DeprecationWarning)
# filterwarnings("ignore",category=UserWarning)


datasets = ['MUTAG','PROTEINS','IMDB-BINARY','IMDB-MULTI','NCI1','PTC_MR']
device = 'cpu' if not is_available() else 'cuda'


def evaluate_dataset(ds_path, dataset: tu_dataset_type, num_folds: int, seed:int):
    run_path = f'runs/tudatasets/{dataset}'
    csv_logger = CSVLogger(save_dir=run_path,name='meta_log')
    param_loop = tqdm(total=8,leave=False)
    best_mean = -1
    best_std = math.nan
    if dataset in ['IMDB-BINARY','PROTEINS','IMDB-MULTI','REDDIT-BINARY']:
        pre_tf = get_pre_transform(dataset,max_cycle_size=10)
    else:
        pre_tf = get_pre_transform(dataset)
        
    for batch_size in [32,128]:
        data_handler = DataHandler(ds_path,device,batch_size,1000,1000,dataset,pre_tf,get_transform(dataset),num_folds,seed)
        for hidden_channels in [16,32]:
            for lr in [0.01,0.001]:
                csv_logger.log_metrics({
                    'batch_size' : batch_size,
                    'hidden_channels' : hidden_channels,
                    'lr' : lr,
                })
                csv_logger.save()
                scores = []
                loop = tqdm(range(num_folds),position=1,total=num_folds,leave=False)
                for fold_idx in loop:
                    data_handler.set_fold_idx(fold_idx)
                    model = Net(hidden_channels,4,0,dataset,'sum',0.00001,0.1,'sum').to(device)
                    model = ModelHandler(model,lr,dataset,'adam',1,lr_patience=50, mode='max', cooldown=0)
                    trainer, version, inner_logger = get_trainer(run_path,350,1E-5,'max',2,True)#type: ignore
                    inner_logger.log_hyperparams({
                        'batch_size' : batch_size,
                        'hidden_channels' : hidden_channels,
                        'lr' : lr,
                    })
                    trainer.fit(model,datamodule=data_handler)
                    test_result = trainer.test(model,ckpt_path='best',datamodule=data_handler)[0]['test_score']
                    scores.append(test_result)
                    csv_logger.log_metrics({'test_score':test_result,'version':version},fold_idx)
                    csv_logger.save()
                    loop.set_postfix(mean=np.mean(scores),std=np.std(scores))
                mean = np.mean(scores)
                std = np.std(scores)
                csv_logger.log_metrics({'score_mean':mean,'score_std':std},num_folds)#type: ignore
                if mean > best_mean:
                    best_mean = mean
                    best_std = std
                    param_loop.set_postfix(best_mean=best_mean,best_std=best_std)
                param_loop.update()
    csv_logger.log_metrics({'best_score_mean':best_mean,'best_score_std':best_std})#type: ignore

def ensure_exists(path: str):
    base = ''
    for segment in path.split('/'):
        base = f'{base}{segment}/'
        if not os.path.exists(base):
            os.mkdir(base)



for dataset in datasets:
    ensure_exists(f'runs/tudatasets/{dataset}')
    evaluate_dataset('data',dataset,10,0) #type: ignore