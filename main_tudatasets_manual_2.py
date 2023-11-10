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
from torch import no_grad
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from model_handler import get_loss_fn, get_score_fn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from transforms import get_pre_transform
from torchmetrics import Accuracy

# from warnings import filterwarnings
# filterwarnings("ignore",category=DeprecationWarning)
# filterwarnings("ignore",category=UserWarning)

device = 'cpu' if not is_available() else 'cuda'

def get_loop(loader: DataLoader, position: int, min_interval: int):
    return tqdm(loader,'train',len(loader),False,position=position,mininterval=min_interval)

# training components
num_epochs = 350
#

def train_epoch(loader: DataLoader, model: Net, optim: Adam, loss_fn, train_score_fn: Accuracy, position: int, min_interval: float, bar: bool) -> tuple[float,float]:
    # TODO: pre move dataset to cuda.
    if bar:
        loop = get_loop(loader,position, min_interval)
        tot_graphs = 0
    else:
        loop = loader
    tot_loss = 0
    for batch in loader:
        pred = model(batch)
        loss = loss_fn(pred,batch.y)
        train_score_fn(pred,batch.y)
        tot_loss += batch.num_graphs * loss.detach().item()
        if bar:
            tot_graphs += batch.num_graphs#type: ignore
            loop.set_postfix({#type: ignore
                'loss' : tot_loss / tot_graphs,
                'train score' : train_score_fn.eval()
            })

        optim.zero_grad()
        loss.backward()
        optim.step()


    return tot_loss / len(loader), train_score_fn.compute()#type: ignore
@no_grad()
def test(loader: DataLoader, model: Net, score: Accuracy) -> float:
    model.eval()
    for batch in loader:
        score(model(batch),batch.y)
    model.train()
    return score.compute()# type: ignore

def train_model(train_set: DataLoader, val_set: DataLoader, model: Net, ds: tu_dataset_type, train_score_fn: Accuracy, val_score_fn: Accuracy, lr: float, logger: SummaryWriter, position: int, min_interval: float, epoch_bar: bool) -> float:
    model.train()
    loop = tqdm(range(1,num_epochs + 1),'Epoch',num_epochs,False,position=position,mininterval=min_interval)
    optim = Adam(model.parameters(),lr)
    scheduler = StepLR(optim,50,0.5)
    loss_fn = get_loss_fn(ds)
    max_val_score = -1
    for epoch in loop:
        loss, train_score = train_epoch(train_set,model,optim,loss_fn,train_score_fn,position+1,min_interval,epoch_bar)
        logger.add_scalar('loss',loss)
        logger.add_scalar('train score',train_score)
        val_score = test(val_set,model,val_score_fn)
        logger.add_scalar('val score',val_score)
        if val_score > max_val_score:
            loop.set_postfix({'best val':val_score})
        loop.set_postfix({
            'loss' : loss,
            'train score' : train_score,
            'valid score' : val_score,
        })
        scheduler.step()
    return max_val_score

def train_folds(data: DataHandler, dataset: tu_dataset_type, lr: float, batch_size: int, hidden_channels: int, master_logger: CSVLogger, path: str, position: int,min_interval: float, epoch_bar: bool) -> tuple[float,float]:
    master_logger.log_metrics({
        'lr' : lr,
        'batch_size' : batch_size,
        'hidden_channels' : hidden_channels,
    })
    master_logger.save()
    loop = tqdm(range(0,data.num_folds),'Epoch',data.num_folds,False,position=position,mininterval=min_interval)
    data.setup('fit')
    data.ds._data = data.ds._data.to('cuda')#type: ignore
    train_score_fn = get_score_fn(dataset).to('cuda')
    val_score_fn = get_score_fn(dataset).to('cuda')
    scores = []
    for fold in loop:
        data.set_fold_idx(fold)
        logger = SummaryWriter(path,f"lr_{lr}_batchsize_{batch_size}_channels_{hidden_channels}_fold_{fold}")
        score = train_model(
            data.train_dataloader(),
            data.val_dataloader(),
            Net(hidden_channels,4,0.,dataset,'sum',0.00001,0.1,'sum').to(device),
            dataset,
            train_score_fn, #type: ignore
            val_score_fn, #type: ignore
            lr,
            logger,
            position+1,
            min_interval,
            epoch_bar
        )
        scores.append(score)
        master_logger.log_metrics({
            'lr' : lr,
            'batch_size' : batch_size,
            'hidden_channels' : hidden_channels,
            'fold' : fold,
            'score' : score,
        })
        master_logger.save()
    mean, std = float(np.mean(scores)), float(np.std(scores))
    master_logger.log_metrics({
        'lr' : lr,
        'batch_size' : batch_size,
        'hidden_channels' : hidden_channels,
        'mean' : mean,
        'std' : std,
    })
    master_logger.save()
    return mean, std

def evaluate_dataset(dataset: tu_dataset_type, batch_size: int, lr: int, hidden_channels: int, num_trials: int, seed:int, csv_logger: CSVLogger, position:int, min_interval: float, epoch_bar: bool):
    run_path = f'runs/tudatasets/{dataset}'
    pre_tf = get_pre_transform(dataset) if not dataset in ['PROTEINS','IMDB-BINARY','IMDB-MULTI'] else get_pre_transform(dataset,max_cycle_size=10)
    data_handler = DataHandler('data',device,batch_size,1000,1000,dataset,pre_tf,get_transform(dataset),num_trials,seed)
    best_mean = -1
    # best_std = math.nan
    csv_logger.log_metrics({
        'batch_size' : batch_size,
        'hidden_channels' : hidden_channels,
        'lr' : lr,
    })
    
    mean, std = train_folds(data_handler,dataset,lr,batch_size,hidden_channels,csv_logger,run_path,position,min_interval,epoch_bar)
    # if mean > best_mean:
    #     best_mean = mean
    #     best_std = std
    return mean, std

# def ensure_exists(path: str):
#     base = ''
#     for segment in path.split('/'):
#         base = f'{base}{segment}/'
#         if not os.path.exists(base):
#             os.mkdir(base)
param_sets = [
    { # based on old run.
        'dataset' : 'IMDB-BINARY',
        'batch_size' : 128,
        'hidden_channels' : 16,
        'lr' : 0.01,
        'num_trials' : 2,
    },
    { # based on old run.
        'dataset' : 'IMDB-MULTI',
        'batch_size' : 128,
        'hidden_channels' : 16,
        'lr' : 0.001,
        'num_trials' : 2,
    },
    # {
    #     'dataset' : 'MUTAG',
    #     'batch_size' : 32,
    #     'hidden_channels' : 16,
    #     'lr' : 0.01,
    #     'num_trials' : 7,
    # },
    { # based on old run.
        'dataset' : 'NCI1',
        'batch_size' : 128,
        'hidden_channels' : 32,
        'lr' : 0.001,
        'num_trials' : 2,
    },
    # TODO: add NCI109
    # TODO: add PROTEINS
    # TODO: add REDDIT-BINARY
]
csv_logger = CSVLogger(save_dir='runs/tudatasets',name='meta_log')

loop = tqdm(param_sets,'Param Set',total=len(param_sets))
for params in loop:
    # ensure_exists(f'runs/tudatasets/{dataset}')
    evaluate_dataset(**params,csv_logger=csv_logger,seed=0,position=1,min_interval=1,epoch_bar=False) #type: ignore