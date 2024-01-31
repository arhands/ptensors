# from model_gine_2 import Net
from torch.cuda import is_available
from argparse import ArgumentParser
from datetime import datetime
from utils import get_run_path
from model_handler import ModelHandler
from data_handler import ZINCDatasetHandler
from train_handler import get_trainer
from model import Net
import os

parser = ArgumentParser()
parser.add_argument('--hidden_channels',type=int,default=128)
parser.add_argument('--num_layers',type=int,default=4)
parser.add_argument('--dropout',type=float,default=0.)

parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--num_epochs',type=int,default=1000)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--min_lr',type=float,default=1E-5)
parser.add_argument('--train_batch_size',type=int,default=128)
parser.add_argument('--cooldown',type=int,default=0)
parser.add_argument('--subset',action='store_false')

parser.add_argument('--eval_batch_size',type=int,default=512)
parser.add_argument('--run_path',type=str,default=None)
parser.add_argument('--force_use_cpu',action='store_true')
parser.add_argument('--use_old_model',action='store_true')
parser.add_argument('--bn_eps',type=float,default=0.00001)
parser.add_argument('--bn_momentum',type=float,default=0.05)
parser.add_argument('--lr_decay',type=float,default=0.5)
parser.add_argument('--ptensor_reduction',type=str,default='mean')
parser.add_argument('--valid_score_sample_size',type=int,default=1)

args = parser.parse_args()

dataset = 'ZINC'

def get_ds_path(old_model: bool, subset: bool):
    if old_model:
        return 'data'
    else:
        assert subset
        return 'data'

device = 'cpu' if args.force_use_cpu or not is_available() else 'cuda'

from data_transforms import AddNodes, AddEdges, AddChordlessCycles, AddTransferMap, GeneratePtensObject
pre_transform = GeneratePtensObject(
    [
        AddNodes(1),
        AddEdges(1),
        AddChordlessCycles(1),
    ],
    [
        AddTransferMap('nodes','edges',0,False),
        AddTransferMap('nodes','cycles',1,False),
        AddTransferMap('edges','cycles',1,True),
    ]
)

model = Net(args.hidden_channels,args.num_layers,args.dropout,'ZINC','sum',args.bn_eps,args.bn_momentum,args.ptensor_reduction).to(device)
def ensure_exists(path: str):
    base = ''
    for segment in path.split('/'):
        base = f'{base}{segment}/'
        if not os.path.exists(base):
            os.mkdir(base)

ds_path = get_ds_path(args.use_old_model,args.subset)
if args.run_path is None:
    run_path = get_run_path('runs')
else:
    run_path = args.run_path
    ensure_exists(run_path)

trainer, version = get_trainer(run_path,args.num_epochs,args.min_lr,'min')

ensure_exists(f"{run_path}/lightning_logs/version_{version}/")
overview_log_path = f"{run_path}/lightning_logs/version_{version}/summary.log"
with open(overview_log_path,'w') as file:
    intital_info = {
        'start date and time' : datetime.now().strftime(r"%d/%m/%Y %H:%M:%S"),
        **vars(args)
    }
    lines = [
        f"{k} : {intital_info[k]}\n\r"
        for k in intital_info
    ]
    file.writelines(lines)

model = ModelHandler(model,args.lr,dataset,'adam',args.valid_score_sample_size,lr_patience = args.patience, mode='min', cooldown=args.cooldown,lr_decay=args.lr_decay)

data_handler = ZINCDatasetHandler(ds_path,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size,pre_transform,None,args.subset)

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