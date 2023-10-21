from typing import Any, Callable, Dict, Literal
import lightning.pytorch as pl
from pytorch_optimizer.base.types import OPTIMIZER, PARAMETERS
from torch.nn import Module
from torch import Tensor
import torch
from objects import MultiScaleData
from torchmetrics import MeanAbsoluteError, MeanSquaredError, RunningMean
from torchmetrics.classification import BinaryAUROC
from torch_geometric.data import Batch
from torch.nn import L1Loss, BCEWithLogitsLoss, MSELoss
from torch.optim import Adam, Optimizer
from pytorch_optimizer.optimizer.sam import SAM
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

class Log10MSE(MeanSquaredError):
    def compute(self) -> Tensor:
        return super().compute().log10()

def get_loss_fn(ds: Literal['ZINC','ogbg-molhiv','ogbg-moltox21','peptides-struct','graphproperty']):
    if ds in ['ZINC','peptides-struct']:
        return L1Loss()
    elif ds in ['ogbg-molhiv','ogbg-moltox21']:
        return BCEWithLogitsLoss()
    elif ds == 'graphproperty':
        return MSELoss()
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')

def get_score_fn(ds: Literal['ZINC','ogbg-molhiv','ogbg-moltox21','peptides-struct','graphproperty']):
    if ds in ['ZINC','peptides-struct']:
        return MeanAbsoluteError()
    elif ds in ['ogbg-molhiv','ogbg-moltox21']:
        return BinaryAUROC()
    elif ds == 'graphproperty':
        return Log10MSE()
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')

def get_lr_scheduler(ds: Literal['ZINC','ogbg-molhiv','graphproperty','ogbg-moltox21','peptides-struct'], optimizer: Optimizer, **args):
    if ds in ['ZINC','peptides-struct']:
        return {
            "scheduler" : ReduceLROnPlateau(optimizer,args['mode'],0.5,args['lr_patience'],cooldown=args['cooldown'],verbose=True),
            "monitor" : "val_score_multi"
        }
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
    def __init__(self, model: Module, lr: float, ds: Literal['ZINC','ogbg-molhiv','ogbg-moltox21','peptides-struct','graphproperty'], optimizer: Literal['adam','asam'], running_mean_window_size: int = 1, **lr_schedular_args) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.loss_fn = get_loss_fn(ds)
        self.valid_score_fn = get_score_fn(ds)
        self.test_score_fn = get_score_fn(ds)
        self.running_mean_score_fn = RunningMean(running_mean_window_size)
        self.lr = lr

        self.lr_scheduler_args = lr_schedular_args
        self.ds : Literal['ZINC','ogbg-molhiv','graphproperty','ogbg-moltox21','peptides-struct'] = ds
        self.optimizer_name : Literal['adam','asam'] = optimizer

        # if optimizer == 'asam':
        #     self.automatic_optimization = False

    def forward(self, data: MultiScaleData) -> Tensor:
        return self.model(data)#.flatten()
    
    def training_step(self, batch: Batch, batch_idx: int):
        pred = self(batch)
        y = batch.y
        if self.ds == 'ogbg-moltox21':
            mask = ~torch.isnan(y)
            pred = pred[mask]
            y = y[mask]
        loss = self.loss_fn(pred,y)
        self.log('train_loss',loss,True,batch_size=batch.num_graphs,on_step=False,on_epoch=True)

        # if self.optimizer_name == 'asam':
        #     self.manual_backward(loss,retain_graph=True)
        #     opt : SAM = self.optimizers(False) #type: ignore
        #     opt.first_step(True)

        #     self.manual_backward(loss)
        #     self.loss_fn(pred,batch.y)
        #     opt.second_step(True)
        return loss
    def validation_step(self, batch: Batch, batch_idx: int):
        self.valid_score_fn(self(batch),batch.y)
    def test_step(self, batch: Batch, batch_idx: int):
        self.test_score_fn(self(batch),batch.y)
    def on_validation_epoch_end(self) -> None:
        self.log('lr-Adam',self.optimizers(False).param_groups[0]['lr'])
        valid_score = self.valid_score_fn.compute()
        self.valid_score_fn.reset()
        self.log('val_score',valid_score,on_epoch=True,prog_bar=True)
        self.running_mean_score_fn.update(valid_score)
        self.log('val_score_multi',self.running_mean_score_fn.compute(),on_epoch=True)
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