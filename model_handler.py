import lightning.pytorch as pl
from torch.nn import Module
from torch import Tensor
from objects import MultiScaleData
from torchmetrics import MeanAbsoluteError
from torch_geometric.data import Batch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ModelHandler(pl.LightningModule):
    def __init__(self, model: Module, lr: float, lr_patience: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = L1Loss()
        self.valid_score_fn = MeanAbsoluteError()
        self.test_score_fn = MeanAbsoluteError()
        self.lr = lr
        self.lr_patience = lr_patience

    def forward(self, data: MultiScaleData) -> Tensor:
        return self.model(data).flatten()
    
    def training_step(self, batch: Batch, batch_idx: int):
        pred = self(batch)
        loss = self.loss_fn(pred,batch.y.flatten())
        self.log('train_loss',loss,True,batch_size=batch.num_graphs,on_step=True,on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        self.valid_score_fn(self(batch),batch.y.flatten())
    def test_step(self, batch: Batch, batch_idx: int):
        self.test_score_fn(self(batch),batch.y.flatten())
    def on_validation_epoch_end(self) -> None:
        self.log('val_score',self.valid_score_fn,on_epoch=True)
    def on_test_epoch_end(self) -> None:
        self.log('test_score',self.test_score_fn,on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),lr=self.lr)
        scheduler = {
            "scheduler" : ReduceLROnPlateau(optimizer,'min',0.5,self.lr_patience),
            "monitor" : "val_score"
        }
        return {
            "optimizer" : optimizer,
            "lr_scheduler" : scheduler 
        }