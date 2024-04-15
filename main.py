from argparse import Namespace

import wandb
from arg_handler import get_args
args: Namespace = get_args()

from typing import Literal

from torch.cuda import is_available
from model_handler import ModelHandler

from data_handler import DataHandler, dataset_type, _tu_datasets, get_data_handler
from train_handler import get_trainer
from data_transforms import AddNodes, AddEdges, AddChordlessCycles, AddTransferMap, GeneratePtensObject
from lightning.pytorch.loggers.wandb import WandbLogger
from torch_geometric import seed_everything
from utils import ensure_exists
seed_everything(args.seed)

# TODO: come up for better solution for this.
import warnings
import logging
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps if you want to see logs for the training epoch.*")
warnings.filterwarnings("ignore", ".*PU available:.*",module="lightning.pytorch")
warnings.filterwarnings("ignore", ".*is an invalid version and will not be supported in a future release*",module="pkg_resources/__init__")

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

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
ds_name : str = args.dataset
seed: int = args.seed
ds_path = './data/'
run_path: str = f'./runs/{ds_name}/{seed}/'
ensure_exists(run_path + "wandb")

dataset: dataset_type
train_batch_size = args.batch_size
hidden_channels = args.hidden_channels
lr = args.hidden_channels
mode: Literal['min', 'max'] = args.mode

logger = WandbLogger(f"{args.dataset}-{args.seed}",run_path)
wandb.define_metric("val_score",summary=mode)
wandb.define_metric("test_score",summary=mode)

data_handler: DataHandler = get_data_handler(pre_transform,args)
data_handler.setup('fit')
model: ModelHandler = ModelHandler.from_data_handler_ds(data_handler,args)
trainer = get_trainer(run_path,args.num_epochs,args.min_lr,mode,logger,args.show_epoch_progress_bar,args.enable_model_summary)
trainer.fit(model,datamodule=data_handler)
trainer.test(model,ckpt_path='best',datamodule=data_handler)

