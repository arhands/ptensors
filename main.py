from model_3 import Net
# from model_gine_2 import Net
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
parser.add_argument('--num_layers',type=int,default=4)
parser.add_argument('--residual',action='store_true')
parser.add_argument('--dropout',type=float,default=0.)

parser.add_argument('--patience',type=int,default=20)
parser.add_argument('--num_epochs',type=int,default=1000)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--min_lr',type=float,default=1E-5)
parser.add_argument('--train_batch_size',type=int,default=128)

parser.add_argument('--eval_batch_size',type=int,default=512)
parser.add_argument('--run_path',type=str,default=None)
parser.add_argument('--force_use_cpu',action='store_true')

args = parser.parse_args()

def ensure_exists(path: str):
    base = ''
    for segment in path.split('/'):
        base = f'{base}{segment}/'
        if not os.path.exists(base):
            os.mkdir(base)

ds_path = 'data/ZINC'
if args.run_path is None:
    run_path = get_run_path('runs')
else:
    run_path = args.run_path
    ensure_exists(run_path)


overview_log_path = f"{run_path}/summary.log"
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

device = 'cpu' if args.force_use_cpu or not is_available() else 'cuda'

model = Net(args.hidden_channels,args.num_layers,args.dropout,'ZINC',args.residual).to(device)

model = ModelHandler(model,args.lr,args.patience)

data_handler = DataHandler(ds_path,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size)

with open(overview_log_path,'a') as file:
    file.writelines([
        'Model Summary',
        str(model)
    ])

trainer = get_trainer(run_path,args.num_epochs,args.min_lr)

trainer.fit(model,datamodule=data_handler)

test_result = trainer.test(model,ckpt_path='best',datamodule=data_handler)

with open(overview_log_path,'a') as file:
    file.writelines([
        'test result',
        str(test_result)
    ])