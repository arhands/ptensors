# from model_gine_2 import Net
import math
from torch.cuda import is_available
from argparse import ArgumentParser
from datetime import datetime
from utils import get_run_path
from model_handler import ModelHandler
from data_handler import DataHandler
from train_handler import get_trainer
import os

# from warnings import filterwarnings
# filterwarnings("ignore",category=DeprecationWarning)
# filterwarnings("ignore",category=UserWarning)

parser = ArgumentParser()
parser.add_argument('--hidden_channels',type=int,default=128)
parser.add_argument('--num_layers',type=int,default=5)
parser.add_argument('--dropout',type=float,default=0.0)

parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--cooldown',type=int,default=0)
parser.add_argument('--num_epochs',type=int,default=1000)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--min_lr',type=float,default=1E-5)
parser.add_argument('--train_batch_size',type=int,default=128)

parser.add_argument('--eval_batch_size',type=int,default=1000)
parser.add_argument('--window',type=int,default=1)
parser.add_argument('--max_ring_size',type=int,default=math.inf)
parser.add_argument('--run_path',type=str,default=None)
parser.add_argument('--force_use_cpu',action='store_true')
parser.add_argument('--use_old_model',action='store_true')

parser.add_argument('--bn_eps',type=float,default=0.00001)
parser.add_argument('--bn_momentum',type=float,default=0.1)
parser.add_argument('--ptensor_reduction',type=str,default='mean')
parser.add_argument('--optimizer',type=str,default='adam')
parser.add_argument('--cyclecycle',action='store_true')

args = parser.parse_args()

dataset_name = 'peptides-struct'
device = 'cpu' if args.force_use_cpu or not is_available() else 'cuda'
from transforms import get_pre_transform, get_transform

ds_path = 'data' if not args.cyclecycle else 'data_c2c'
from model import Net
model = Net(
    args.hidden_channels,
    args.num_layers,
    args.dropout,
    dataset_name,
    'mean',
    args.bn_eps,
    args.bn_momentum,
    args.ptensor_reduction,args.cyclecycle).to(device)


def ensure_exists(path: str):
    base = ''
    for segment in path.split('/'):
        base = f'{base}{segment}/'
        if not os.path.exists(base):
            os.mkdir(base)

if args.run_path is None:
    run_path = get_run_path('runs')
else:
    run_path = args.run_path
    ensure_exists(run_path)

trainer, version = get_trainer(run_path,args.num_epochs,args.min_lr,'min')

overview_log_path = f"{run_path}/summary_{version}.log"
with open(overview_log_path,'a') as file:
    intital_info = {
        'start date and time' : datetime.now().strftime(r"%d/%m/%Y %H:%M:%S"),
        **vars(args)
    }
    lines = [
        f"{k} : {intital_info[k]}\n\r"
        for k in intital_info
    ]
    file.writelines(lines)



model = ModelHandler(model,args.lr,dataset_name,args.optimizer,args.window,mode='min',cooldown=args.cooldown,lr_patience=args.patience)

data_handler = DataHandler(ds_path,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size,dataset_name,get_pre_transform(dataset_name,False,args.max_ring_size,args.cyclecycle),get_transform(dataset_name))

with open(overview_log_path,'a') as file:
    file.writelines([
        'Model Summary',
        str(model)
    ])


trainer.fit(model,datamodule=data_handler)

test_result = trainer.test(model,ckpt_path='best',datamodule=data_handler)

with open(overview_log_path,'a') as file:
    file.writelines([
        'test result',
        str(test_result)
    ])