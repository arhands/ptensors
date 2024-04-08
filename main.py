from argparse import ArgumentParser
from math import inf, nan
from typing import Literal, NamedTuple
import numpy as np
from tqdm import tqdm
parser = ArgumentParser()
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--hidden_channels',type=str,default="128")
# parser.add_argument('--hidden_channels',type=int,default=128)
parser.add_argument('--num_layers',type=int,default=4)
parser.add_argument('--dropout',type=float,default=0.)

parser.add_argument('--patience',type=int,default=30)
parser.add_argument('--num_epochs',type=int,default=1000)
parser.add_argument('--lr',type=str,default="0.001")
# parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--min_lr',type=float,default=1E-5)
parser.add_argument('--train_batch_size',type=str,default="128")
# parser.add_argument('--train_batch_size',type=int,default=128)

parser.add_argument('--eval_batch_size',type=int,default=512)
parser.add_argument('--run_path',type=str,default=None)
parser.add_argument('--bn_eps',type=float,default=0.00001)
parser.add_argument('--bn_momentum',type=float,default=0.1)
parser.add_argument('--lr_decay',type=float,default=0.5)
parser.add_argument('--ptensor_reduction',type=str,default='mean')
parser.add_argument('--readout',type=str,default='sum')
parser.add_argument('--device',type=str,default=None)
parser.add_argument('--include_cycle2cycle',action='store_true')
parser.add_argument('--optimizer',type=str,default='adam')
parser.add_argument('--num_trials',type=int,default=10)
parser.add_argument('--max_cycle_size',type=int,default=None)
parser.add_argument('--seed',type=int,default=None)
parser.add_argument('--enable_model_summary',action="store_true")
parser.add_argument('--show_epoch_progress_bar',action="store_true")

parser.add_argument('--wandb_project_name',type=str,default=None)

args = parser.parse_args()

# logging.getLogger("lightning.pytorch.utilities").addFilter(device_info_filter)
# warnings.filterwarnings("ignore", ".*available:.*")
# warnings.filterwarnings("ignore", "*is an invalid version and will not be supported in a future release*")


from torch.cuda import is_available
from utils import get_run_path, ensure_exists
from model_handler import ModelHandler, get_mode
from data_handler import dataset_type, _tu_datasets, ZINCDatasetHandler, OGBGDatasetHandler, TUDatasetHandler
from train_handler import get_trainer
from model import Net
from data_transforms import AddNodes, AddEdges, AddChordlessCycles, AddTransferMap, GeneratePtensObject
import torch_geometric
if args.seed is not None:
    torch_geometric.seed_everything(args.seed)

# TODO: come up for better solution for this.
# import warnings
# import logging
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
# warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps if you want to see logs for the training epoch.*")
# warnings.filterwarnings("ignore", ".*PU available:.*",module="lightning.pytorch")
# warnings.filterwarnings("ignore", ".*is an invalid version and will not be supported in a future release*",module="pkg_resources/__init__")

# logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

device: str
if args.device is None:
    device = 'cuda' if is_available() else 'cpu'
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
        AddChordlessCycles(1,args.max_cycle_size),
    ],
    transfer_maps
)

def get_arg_combinations(keys: list[str]) -> list[dict[str,str]]:
    if len (keys) == 0:
        return [dict()]
    else:
        key = keys.pop()
        children: list[dict[str,str]] = get_arg_combinations(keys)
        valueStr: str = getattr(args,key)
        values: list[str]
        if '[' not in valueStr or ']' not in valueStr:
            values = [valueStr]
        else:
            valueStr = valueStr[valueStr.index('[') + 1:valueStr.index(']')]
            values = valueStr.split(',')
        return [
            {
                key : value,
                **child
            }
            for value in values
            for child in children
        ]



ds_path = './data/'
if args.run_path is None:
    run_path = get_run_path('runs')
else:
    run_path = args.run_path
    ensure_exists(run_path)
argCombinations: list[dict[str, str]] = get_arg_combinations(['train_batch_size','hidden_channels','lr'])
datasetOptions: list[str] = [
    v['dataset']
    for v in get_arg_combinations(['dataset'])
]
bestArgs = dict()
dataset: dataset_type
dsLoop = tqdm(datasetOptions,"ds",len(datasetOptions))
for dataset in datasetOptions:#type: ignore
    dsLoop.set_postfix(cur_ds=dataset)
    bestDatasetArgs = dict()
    bestDatasetScore : float | int = nan
    argChoiceLoop = tqdm(argCombinations,'args',len(argCombinations),False,position=1)
    # for argChoices in argCombinations:
    for argChoices in argChoiceLoop:
        train_batch_size = int(argChoices['train_batch_size'])
        hidden_channels = int(argChoices['hidden_channels'])
        lr = float(argChoices['lr'])
        argChoiceLoop.set_postfix(
            batch_size = train_batch_size,
            channels = hidden_channels,
            lr = lr
        )

        mode: Literal['min', 'max'] = get_mode(dataset)

        model = Net(hidden_channels,args.num_layers,args.dropout,dataset,args.readout,0.00001,args.bn_momentum,args.ptensor_reduction,args.include_cycle2cycle).to(device)



        data_handler : ZINCDatasetHandler|OGBGDatasetHandler|TUDatasetHandler
        handlerArgs = {
            'root' : ds_path,
            'device' : device,
            'train_batch_size' : train_batch_size,
            'val_batch_size' : args.eval_batch_size,
            'test_batch_size' : args.eval_batch_size,
            'pre_transform' : pre_transform,
        }
        if dataset in ['ZINC','ZINC-Full']:
            data_handler = ZINCDatasetHandler(subset = dataset != 'ZINC-Full',**handlerArgs)
            dataset = 'ZINC'
        elif dataset in ['ogbg-molhiv','ogbg-moltox21']:
            data_handler = OGBGDatasetHandler(ds_name=dataset,**handlerArgs)#type: ignore
        elif dataset in _tu_datasets:
            data_handler = TUDatasetHandler(ds_name=dataset,num_folds=args.num_trials,seed=0,**handlerArgs)#type: ignore
        else:
            raise NotImplementedError(dataset)
        
        run_results: list[dict[str,float]] = []
        trialLoop = tqdm(range(args.num_trials),'trial',args.num_trials,leave=False,position=2)
        for trialIdx in trialLoop:
            pos = 3
            show_progress_bar: bool = args.show_epoch_progress_bar
            show_model_summary = trialIdx == 0 and args.enable_model_summary
            trainer, version = get_trainer(run_path,args.num_epochs,args.min_lr,mode,args.wandb_project_name,vars(args),pos,show_progress_bar,show_model_summary=show_model_summary)
            model = ModelHandler(model,lr,dataset,args.optimizer,lr_patience = args.patience, mode=mode, cooldown=0,lr_decay=args.lr_decay)
            trainer.fit(model,datamodule=data_handler)
            test_result: dict[str,float] = trainer.test(model,ckpt_path='best',datamodule=data_handler,verbose=False)[0]#type: ignore
            run_results.append(test_result)
            if dataset in _tu_datasets:
                assert isinstance(data_handler,TUDatasetHandler)#TODO: consider making this a generic property
                data_handler.set_fold_idx(trialIdx)
        class Statistic(NamedTuple):
            mininum: float
            maximum: float
            mean: float
            std: float
            median: float
            def __str__(self) -> str:
                return f"min: {self.mininum}, max: {self.maximum}, median: {self.median}, mean: {self.mean}, std: {self.std}"
        result_stats: dict[str,Statistic]|dict[str,float] = dict()
        if len(run_results) > 1:
            key: str
            for key in run_results[0]:
                values: list[float] = [res[key] for res in run_results]
                result_stats[key] = Statistic(
                    min(values),
                    max(values),
                    np.mean(values).item(),
                    np.std(values).item(),
                    np.median(values).item(),
                )
        else:
            result_stats = run_results[0]

        # saving results regarding "this" run if they were better than the previous ones for this dataset.
        stats: Statistic | float = result_stats['test_score']
        score: float | int = stats if isinstance(stats,(float,int)) else stats.mean
        if "score" not in bestDatasetArgs or mode == 'min' and bestDatasetScore > score or mode == 'max' and bestDatasetScore < score:
            dsLoop.set_postfix(cur_ds=dataset,cur_best_score = score)
            bestDatasetScore = score
            bestDatasetArgs = {
                'stats' : stats,
                **argChoices
            }
    bestArgs[dataset] = bestDatasetArgs
summary_lines: list[str] = []
def log(val: str = ""):
    print(val)
    summary_lines.append(val + "\n")
log("Summary of results:")
for ds in bestArgs:
    log(f"\t{ds}:")
    dsBestArgs: dict[str,float] = bestArgs[ds]
    for key in dsBestArgs:
        log(f"\t\t{key}: {dsBestArgs[key]}")
    log()

with open(run_path + "results.log",'w') as F:
    F.writelines(summary_lines)