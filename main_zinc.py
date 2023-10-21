# from model_gine_2 import Net
from torch.cuda import is_available
from argparse import ArgumentParser
from datetime import datetime
from utils import get_run_path
from model_handler import ModelHandler
from data_handler import DataHandler
from train_handler import get_trainer
import os
import optuna
from optuna import load_study, Trial
from transforms import get_pre_transform, get_transform
from model import Net

# from warnings import filterwarnings
# filterwarnings("ignore",category=DeprecationWarning)
# filterwarnings("ignore",category=UserWarning)

parser = ArgumentParser()
parser.add_argument('--dropout',type=float,default=0.)

parser.add_argument('--num_epochs',type=int,default=1000)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--min_lr',type=float,default=1E-5)
parser.add_argument('--train_batch_size',type=int,default=128)

parser.add_argument('--eval_batch_size',type=int,default=512)
parser.add_argument('--run_path',type=str,default=None)
parser.add_argument('--force_use_cpu',action='store_true')
parser.add_argument('--cooldown',type=int,default=0)
parser.add_argument('--bn_eps',type=float,default=0.00001)
parser.add_argument('--ptensor_reduction',type=str,default='mean')
parser.add_argument('--no_cycle2cycle',action='store_true')
parser.add_argument('--valid_score_sample_size',type=int,default=1)

args = parser.parse_args()
ds_path = 'data/ZINC' if args.no_cycle2cycle else 'data/ZINC_c2c'

dataset = 'ZINC'

device = 'cpu' if args.force_use_cpu or not is_available() else 'cuda'


def evaluate(trial: Trial):
    hidden_channels = trial.suggest_int('hidden_channels',32,96,32)
    num_layers = trial.suggest_int('num_layers',4,5)
    bn_momentum = 0.1*2**-trial.suggest_int('bn_momentum',1,2)
    patience = trial.suggest_int('patience',30,30)
    model = Net(
        hidden_channels,
        num_layers,
        args.dropout,'ZINC','sum',args.bn_eps,
        bn_momentum,args.ptensor_reduction,not args.no_cycle2cycle).to(device)
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

    model = ModelHandler(model,args.lr,dataset,'adam',args.valid_score_sample_size,lr_patience = patience, mode='min', cooldown=args.cooldown)

    data_handler = DataHandler(ds_path,device,args.train_batch_size,args.eval_batch_size,args.eval_batch_size,'ZINC',get_pre_transform(dataset,False,include_cycles2cycles=not args.no_cycle2cycle),get_transform(dataset))

    with open(overview_log_path,'a') as file:
        file.writelines([
            'Model Summary',
            str(model)
        ])

    trainer = get_trainer(run_path,args.num_epochs,args.min_lr,'min')

    trainer.fit(model,datamodule=data_handler)

    test_result = trainer.test(model,ckpt_path='best',datamodule=data_handler)

    with open(overview_log_path,'a') as file:
        file.writelines([
            'test result',
            str(test_result)
        ])
    return test_result[0]['test_score']

study = load_study(
    sampler=optuna.samplers.TPESampler(n_ei_candidates=4),
    study_name='ZINC-C2C',
    storage='sqlite:///runs/optuna.db')

study.optimize(evaluate,100)