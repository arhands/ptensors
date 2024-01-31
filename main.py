from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--hidden_channels',type=int,default=128)
parser.add_argument('--num_layers',type=int,default=4)
parser.add_argument('--dropout',type=float,default=0.)

parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--num_epochs',type=int,default=1000)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--min_lr',type=float,default=1E-5)
parser.add_argument('--train_batch_size',type=int,default=128)

parser.add_argument('--eval_batch_size',type=int,default=512)
parser.add_argument('--run_path',type=str,default=None)
parser.add_argument('--force_use_cpu',action='store_true')
parser.add_argument('--bn_eps',type=float,default=0.00001)
parser.add_argument('--bn_momentum',type=float,default=0.1)
parser.add_argument('--lr_decay',type=float,default=0.5)
parser.add_argument('--ptensor_reduction',type=str,default='mean')
parser.add_argument('--readout',type=str,default='sum')
parser.add_argument('--device',type=str,default=None)
parser.add_argument('--include_cycle2cycle',action='store_true')

args = parser.parse_args()

from torch.cuda import is_available
from datetime import datetime
from utils import get_run_path
from model_handler import ModelHandler
from data_handler import dataset_type, _tu_datasets, ZINCDatasetHandler, DataHandler, OGBGDatasetHandler, TUDatasetHandler
from train_handler import get_trainer
from model import Net
import os
from data_transforms import AddNodes, AddEdges, AddChordlessCycles, AddTransferMap, GeneratePtensObject




dataset: dataset_type = args.dataset

device: str
if args.device is None:
    device = 'cpu' if args.force_use_cpu or not is_available() else 'cuda'
else:
    device = args.device

transfer_maps: list[AddTransferMap] = [
        AddTransferMap('nodes','edges',0,False),
        AddTransferMap('nodes','cycles',1,False),
        AddTransferMap('edges','cycles',1,True),
    ]
if args.include_cycle2cycle:
    transfer_maps.append(AddTransferMap('cycles','cycles',1,False))
pre_transform = GeneratePtensObject(
    [
        AddNodes(1),
        AddEdges(1),
        AddChordlessCycles(1),
    ],
    transfer_maps
)

model = Net(args.hidden_channels,args.num_layers,args.dropout,dataset,args.readout,0.00001,args.bn_momentum,args.ptensor_reduction,args.include_cycle2cycle).to(device)
def ensure_exists(path: str):
    base = ''
    for segment in path.split('/'):
        base = f'{base}{segment}/'
        if not os.path.exists(base):
            os.mkdir(base)

ds_path = './data/'
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

model = ModelHandler(model,args.lr,dataset,'adam',1,lr_patience = args.patience, mode='min', cooldown=0,lr_decay=args.lr_decay)

data_handler : DataHandler
if dataset in ['ZINC','ZINC-Full']:
    data_handler = ZINCDatasetHandler(ds_path,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size,pre_transform,None,dataset != 'ZINC-Full')
elif dataset in ['ogbg-molhiv','ogbg-moltox21']:
    data_handler = OGBGDatasetHandler(ds_path,dataset,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size,pre_transform,None)#type: ignore
elif dataset in _tu_datasets:
    data_handler = TUDatasetHandler(ds_path,dataset,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size,pre_transform,None)#type: ignore
else:
    raise NotImplementedError(dataset)

trainer.fit(model,datamodule=data_handler)

test_result = trainer.test(model,ckpt_path='best',datamodule=data_handler)