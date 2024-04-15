from __future__ import annotations
from typing import Callable, Literal, cast, overload, Optional
import lightning.pytorch as pl
from pytorch_optimizer.base.types import OPTIMIZER, PARAMETERS
from torch.nn import Module
from torch import Tensor
import torch
from data import FancyDataObject, PtensObjects
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy
from torch.nn import L1Loss, BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.optim import Adam, Optimizer
from pytorch_optimizer.optimizer.sam import SAM
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from data_handler import DataHandler, dataset_type, tu_dataset_type_list
from torch_geometric.data import InMemoryDataset
from argparse import Namespace
from feature_encoders import get_edge_encoder, get_node_encoder, CycleEmbedding
from model import Net

loss_arg_type_list = ['MAE','BCEWithLogits','MSE','CrossEntropy']
loss_arg_type = Literal['MAE','BCEWithLogits','MSE','CrossEntropy']
def get_loss_fn(name: loss_arg_type) -> Module:
    return {
        'MAE' : L1Loss,
        'BCEWithLogits' : BCEWithLogitsLoss,
        'MSE' : MeanSquaredError,
        'CrossEntropy' : CrossEntropyLoss,
    }[name]()

score_arg_type_list = ['MAE','AUROC','Accuracy','Multi-Label-Accuracy']
score_arg_type = Literal['MAE','AUROC','Accuracy','Multi-Label-Accuracy']
@overload
def get_score_fn(name: Literal['Accuracy','Multi-Label-Accuracy'], num_args: int) -> Metric:...
@overload
def get_score_fn(name: Literal['Accuracy'], num_args: Literal[None] = None) -> BinaryAccuracy:...
@overload
def get_score_fn(name: Literal['MAE','AUROC'], num_args: Literal[None] = None) -> Metric:...
def get_score_fn(name: score_arg_type, num_args: Optional[int] = None) -> Metric:
    if name == 'Accuracy':
        if num_args is None or num_args == 2:
            return BinaryAccuracy()
        else:
            assert num_args > 2
            return MulticlassAccuracy(num_args,average='micro')
    elif name == 'Multi-Label-Accuracy':
        num_args = cast(int,num_args)
        assert num_args > 1
        return MultilabelAccuracy(num_args,average='micro')
    else:
        return {
            'MAE' : MeanAbsoluteError,
            'AUROC' : BinaryAUROC,
        }[name]()

lr_scheduler_arg_type_list = ['ReduceLROnPlateau','StepLR']
lr_scheduler_arg_type = Literal['ReduceLROnPlateau','StepLR']
@overload
def get_lr_scheduler(sched: Literal[None], optimizer: Optimizer, **args) -> None:...
@overload
def get_lr_scheduler(sched: lr_scheduler_arg_type, optimizer: Optimizer, **args) -> Optimizer:...
def get_lr_scheduler(sched: lr_scheduler_arg_type|None, optimizer: Optimizer, **args) -> Optimizer|None:
    return {
        'ReduceLROnPlateau' : lambda : {
                "scheduler" : ReduceLROnPlateau(optimizer,args['mode'],0.5,args['patience'],verbose=True),
                "monitor" : "val_score"
            },
        'StepLR' : StepLR(optimizer,50,0.5),
        None : lambda : None,
    }[sched]()

class ASAM(SAM):
    def __init__(self, params: PARAMETERS, base_optimizer: OPTIMIZER, rho: float = 0.05, **kwargs):
        super().__init__(params, base_optimizer, rho, True, **kwargs)
    
    def step(self, closure: Callable[[],Tensor]):#type: ignore
        value = closure()
        super().step(closure)#type: ignore
        return value


class ModelHandler(pl.LightningModule):
    def __init__(self, model: Module, args: Namespace, num_classes: Optional[int]) -> None:
        super().__init__()
        # self.save_hyperparameters(args,ignore=[
        #     'enable_model_summary',
        #     'show_epoch_progress_bar',
        #     'device',
        #     'eval_batch_size',
        #     'task_type',
        # ])
        self.model = model
        self.loss_fn = get_loss_fn(args.loss)
        self.train_score_fn = get_score_fn(args.eval_metric,num_classes)
        self.valid_score_fn = get_score_fn(args.eval_metric,num_classes)
        self.test_score_fn = get_score_fn(args.eval_metric,num_classes)
        self.lr = args.lr

        self.lr_scheduler_args = {
            "mode" : args.mode,
            "patience" : args.patience,
            "scheduler" : args.lr_scheduler,
            
        }
        self.ds : dataset_type = args.dataset
        self.optimizer_name : Literal['adam','asam'] = args.optimizer

    def forward(self, batch: FancyDataObject, ptens_obj: PtensObjects) -> Tensor:#type: ignore
        return self.model(batch,ptens_obj)
    
    def training_step(self, batch: tuple[FancyDataObject,PtensObjects], batch_idx: int):#type: ignore
        pred = self(*batch)
        y = batch[0].y
        if self.ds == 'ogbg-moltox21':
            mask = ~torch.isnan(y)
            pred = pred[mask]
            y = y[mask]
        loss = self.loss_fn(pred,y)
        self.train_score_fn(pred,y)
        self.log('train_loss',loss,True,batch_size=len(y),on_step=False,on_epoch=True)
        self.log('train_score',self.train_score_fn,True,batch_size=len(y),on_step=False,on_epoch=True)

        return loss
    def validation_step(self, batch: tuple[FancyDataObject,PtensObjects], batch_idx: int):#type: ignore
        self.valid_score_fn(self(*batch),batch[0].y)
    def test_step(self, batch: tuple[FancyDataObject,PtensObjects], batch_idx: int):#type: ignore
        self.test_score_fn(self(*batch),batch[0].y)
    def on_validation_epoch_end(self) -> None:
        self.log('lr-Adam',self.optimizers(False).param_groups[0]['lr'])#type: ignore
        self.log('val_score',self.valid_score_fn,on_epoch=True,prog_bar=True)
    def on_test_epoch_end(self) -> None:
        self.log('test_score',self.test_score_fn,on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = Adam(self.model.parameters(),lr=self.lr)
        elif self.optimizer_name == 'asam':
            optimizer = ASAM(self.model.parameters(),Adam,lr=self.lr)
        else:
            raise NotImplementedError(f'Optimizer \"{self.optimizer_name}\" not handled.')
        scheduler = get_lr_scheduler(self.lr_scheduler_args["scheduler"],optimizer,**self.lr_scheduler_args)#type: ignore
        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler 
            }
    @classmethod
    def from_data_handler_ds(cls, data: DataHandler, args: Namespace) -> ModelHandler:
        ltype = data.ltype
        if ltype == 'single-dim':
            out_channels = 1
        else:
            if not hasattr(data,'splits'):
                data.setup('fit')
            ds: InMemoryDataset = data.splits['train']
            if ltype == 'multi-class':
                out_channels = int(ds.y.max()) + 1
            else: # if ltype == 'multi-label'
                out_channels = ds.y.size(1)
            assert out_channels > 2
        net: Net = Net.from_in_memory_ds(ds,out_channels,args)
        return cls(net,args,out_channels)