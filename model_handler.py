from typing import Any, Callable, Dict, Literal
import lightning.pytorch as pl
from pytorch_optimizer.base.types import OPTIMIZER, PARAMETERS
from torch.nn import Module
from torch import Tensor
import torch
from data import PtensObjects
from torchmetrics import MeanAbsoluteError, MeanSquaredError, RunningMean
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, MulticlassAccuracy
from torch_geometric.data import Batch
from torch.nn import L1Loss, BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.optim import Adam, Optimizer
from pytorch_optimizer.optimizer.sam import SAM
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from data_handler import dataset_type, _tu_datasets

class Log10MSE(MeanSquaredError):
    def compute(self) -> Tensor:
        return super().compute().log10()

def get_mode(ds: dataset_type) -> Literal['min','max']:
    if ds in ['ZINC','ZINC-Full','peptides-struct']:
        return 'min'
    else:
        return 'max'

def get_loss_fn(ds: dataset_type):
    if ds in ['ZINC','peptides-struct']:
        return L1Loss()
    elif ds in ['ogbg-molhiv','ogbg-moltox21','MUTAG','PROTEINS','IMDB-BINARY','REDDIT-BINARY','NCI1','NCI109','PTC_MR']:
        return BCEWithLogitsLoss()
    elif ds == 'graphproperty':
        return MSELoss()
    elif ds in ['ENZYMES','COLLAB','IMDB-MULTI']:
        return CrossEntropyLoss()
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')

def get_score_fn(ds: dataset_type):
    if ds in ['ZINC','peptides-struct']:
        return MeanAbsoluteError()
    elif ds in ['ogbg-molhiv','ogbg-moltox21']:
        return BinaryAUROC()
    elif ds == 'graphproperty':
        return Log10MSE()
    elif ds in ['MUTAG','PROTEINS','IMDB-BINARY','REDDIT-BINARY','NCI1','NCI109','PTC_MR']:
        return BinaryAccuracy()
    elif ds == 'ENZYMES':
        return MulticlassAccuracy(6,average='micro')
    elif ds in ['COLLAB','IMDB-MULTI']:
        return MulticlassAccuracy(3,average='micro')
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')

def get_lr_scheduler(ds: dataset_type, optimizer: Optimizer, **args):
    if ds in ['ZINC','ZINC-Full','peptides-struct']:
        return {
            "scheduler" : ReduceLROnPlateau(optimizer,args['mode'],args['lr_decay'],args['lr_patience'],cooldown=args['cooldown'],verbose=True),
            "monitor" : "val_score"
        }
    elif ds in _tu_datasets:
        return StepLR(optimizer,50,0.5)
    elif ds in ['ogbg-molhiv','ogbg-moltox21']:
        return None
    elif ds == 'graphproperty':
        return StepLR(optimizer,args['lr_step_size'],args['lr_decay'])
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')


class ASAM(SAM):
    def __init__(self, params: PARAMETERS, base_optimizer: OPTIMIZER, rho: float = 0.05, **kwargs):
        super().__init__(params, base_optimizer, rho, True, **kwargs)
    
    def step(self, closure: Callable[[],Tensor]):
        # def wrapped_closure():
        #     loss : Tensor = closure()
        #     loss.backward()
        #     return loss
        value = closure()
        super().step(closure)#type: ignore
        return value


class ModelHandler(pl.LightningModule):
    def __init__(self, model: Module, lr: float, ds: dataset_type, optimizer: Literal['adam','asam'], running_mean_window_size: int = 1, **lr_schedular_args) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.loss_fn = get_loss_fn(ds)
        self.train_score_fn = get_score_fn(ds)
        self.valid_score_fn = get_score_fn(ds)
        self.test_score_fn = get_score_fn(ds)
        # self.running_mean_score_fn = RunningMean(running_mean_window_size)
        self.lr = lr

        self.lr_scheduler_args = lr_schedular_args
        self.ds : dataset_type = ds
        self.optimizer_name : Literal['adam','asam'] = optimizer

        # if optimizer == 'asam':
        #     self.automatic_optimization = False

    def forward(self, x: Tensor, edge_attr: Tensor, data: PtensObjects) -> Tensor:
        # TODO: remove flatten :(
        return self.model(x,edge_attr,data)
    
    def training_step(self, batch: tuple[Tensor,Tensor,Tensor,PtensObjects], batch_idx: int):
        x: Tensor
        edge_attr: Tensor
        y: Tensor
        data: PtensObjects
        x, edge_attr, y, data = batch
        pred = self(x,edge_attr,data)
        if self.ds == 'ogbg-moltox21':
            mask = ~torch.isnan(y)
            pred = pred[mask]
            y = y[mask]
        loss = self.loss_fn(pred,y)
        self.train_score_fn(pred,y)
        self.log('train_loss',loss,True,batch_size=len(y),on_step=False,on_epoch=True)
        self.log('train_score',self.train_score_fn,True,batch_size=len(y),on_step=False,on_epoch=True)

        # if self.optimizer_name == 'asam':
        #     self.manual_backward(loss,retain_graph=True)
        #     opt : SAM = self.optimizers(False) #type: ignore
        #     opt.first_step(True)

        #     self.manual_backward(loss)
        #     self.loss_fn(pred,batch.y)
        #     opt.second_step(True)
        return loss
    def validation_step(self, batch: tuple[Tensor,Tensor,Tensor,PtensObjects], batch_idx: int):
        x: Tensor
        edge_attr: Tensor
        y: Tensor
        data: PtensObjects
        x, edge_attr, y, data = batch
        self.valid_score_fn(self(x,edge_attr,data),y)
    def test_step(self, batch: tuple[Tensor,Tensor,Tensor,PtensObjects], batch_idx: int):
        x: Tensor
        edge_attr: Tensor
        y: Tensor
        data: PtensObjects
        x, edge_attr, y, data = batch
        self.test_score_fn(self(x,edge_attr,data),y)
    def on_validation_epoch_end(self) -> None:
        self.log('lr-Adam',self.optimizers(False).param_groups[0]['lr'])
        # valid_score = self.valid_score_fn.compute()
        # self.valid_score_fn.reset()
        self.log('val_score',self.valid_score_fn,on_epoch=True,prog_bar=True)
        # self.running_mean_score_fn.update(valid_score)
        # self.log('val_score_multi',self.running_mean_score_fn.compute(),on_epoch=True)
    def on_test_epoch_end(self) -> None:
        self.log('test_score',self.test_score_fn,on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = Adam(self.model.parameters(),lr=self.lr)
        elif self.optimizer_name == 'asam':
            optimizer = ASAM(self.model.parameters(),Adam,lr=self.lr)
        else:
            raise NotImplementedError(f'Optimizer \"{self.optimizer_name}\" not handled.')
        scheduler = get_lr_scheduler(self.ds,optimizer,**self.lr_scheduler_args)
        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler 
            }