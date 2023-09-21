from typing import Any, Literal
import lightning.pytorch as pl
from torch.nn import Module
from torch import Tensor
from objects import MultiScaleData
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import BinaryAUROC
from torch_geometric.data import Batch
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.optim import Adam
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

def get_lr_scheduler(ds: Literal['ZINC','ogbg-molhiv'], optimizer: Adam, **args):
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


class ModelHandler(pl.LightningModule):
    def __init__(self, model: Module, lr: float, ds: Literal['ZINC','ogbg-molhiv'], **lr_schedular_args) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.loss_fn = get_loss_fn(ds)
        self.valid_score_fn = get_score_fn(ds)
        self.test_score_fn = get_score_fn(ds)
        self.lr = lr

        self.lr_scheduler_args = lr_schedular_args
        self.ds : Literal['ZINC','ogbg-molhiv'] = ds

    def forward(self, data: MultiScaleData) -> Tensor:
        return self.model(data).flatten()
    
    def training_step(self, batch: Batch, batch_idx: int):
        pred = self(batch)
        loss = self.loss_fn(pred,batch.y.flatten())
        self.log('train_loss',loss,True,batch_size=batch.num_graphs,on_step=False,on_epoch=True)
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
        optimizer = Adam(self.model.parameters(),lr=self.lr)
        scheduler = get_lr_scheduler(self.ds,optimizer,**self.lr_scheduler_args)
        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler 
            }