from typing import Any, Callable, Dict, Literal
import lightning.pytorch as pl
from pytorch_optimizer.base.types import OPTIMIZER, PARAMETERS
from torch.nn import Module
from torch import Tensor, enable_grad
from objects import MultiScaleData
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import BinaryAUROC
from torch_geometric.data import Batch
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.optim import Adam, Optimizer
from pytorch_optimizer.optimizer.sam import SAM
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_loss_fn(ds: Literal['ZINC','ogbg-molhiv']):
    if ds == 'ZINC':
        return L1Loss()
    elif ds == 'ogbg-molhiv':
        return BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')

def get_score_fn(ds: Literal['ZINC','ogbg-molhiv']):
    if ds == 'ZINC':
        return MeanAbsoluteError()
    elif ds == 'ogbg-molhiv':
        return BinaryAUROC()
    else:
        raise NotImplementedError(f'Not implemented for dataset \"{ds}\".')

def get_lr_scheduler(ds: Literal['ZINC','ogbg-molhiv'], optimizer: Optimizer, **args):
    if ds == 'ZINC':
        print("using ReduceLROnPlateau")
        return {
            "scheduler" : ReduceLROnPlateau(optimizer,args['mode'],0.5,args['lr_patience'],verbose=True),
            "monitor" : "val_score"
        }
    elif ds == 'ogbg-molhiv':
        return None
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
    def __init__(self, model: Module, lr: float, ds: Literal['ZINC','ogbg-molhiv'], optimizer: Literal['adam','asam'], **lr_schedular_args) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.loss_fn = get_loss_fn(ds)
        self.valid_score_fn = get_score_fn(ds)
        self.test_score_fn = get_score_fn(ds)
        self.lr = lr

        self.lr_scheduler_args = lr_schedular_args
        self.ds : Literal['ZINC','ogbg-molhiv'] = ds
        self.optimizer_name : Literal['adam','asam'] = optimizer

        # if optimizer == 'asam':
        #     self.automatic_optimization = False

    def forward(self, data: MultiScaleData) -> Tensor:
        return self.model(data).flatten()
    
    def training_step(self, batch: Batch, batch_idx: int):
        pred = self(batch)
        loss = self.loss_fn(pred,batch.y)
        self.log('train_loss',loss,True,batch_size=batch.num_graphs,on_step=False,on_epoch=True)

        # if self.optimizer_name == 'asam':
        #     self.manual_backward(loss,retain_graph=True)
        #     opt : SAM = self.optimizers(False) #type: ignore
        #     opt.first_step(True)

        #     self.manual_backward(loss)
        #     self.loss_fn(pred,batch.y)
        #     opt.second_step(True)
        return loss
    def on_train_start(self) -> None:
        self.log('lr-Adam',self.lr)
    def validation_step(self, batch: Batch, batch_idx: int):
        self.valid_score_fn(self(batch),batch.y.flatten())
    def test_step(self, batch: Batch, batch_idx: int):
        self.test_score_fn(self(batch),batch.y.flatten())
    def on_validation_epoch_end(self) -> None:
        self.log('lr-Adam',self.optimizers(False).param_groups[0]['lr'])
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
        scheduler = get_lr_scheduler(self.ds,optimizer,**self.lr_scheduler_args)
        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler 
            }