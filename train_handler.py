import os
from typing import Literal, Optional
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from lightning import Trainer

# def get_dataset_mode(ds: Literal['ZINC','ogbg-molhiv']):
#     return {
#         'ZINC' : 'min',
#         'ogbg-molhiv' : 'max',
#     }[ds]

def get_trainer(root_dir: str, max_epochs: int, min_lr: Optional[float], mode: Literal['min','max'],pos=0,return_logger=False) -> tuple[Trainer,int]:
    callbacks = [
        ModelCheckpoint(root_dir + '/checkpoints/',monitor='val_score',mode=mode,save_top_k=1),
        ModelSummary(4),
        TQDMProgressBar(0,pos),
    ]
    if min_lr is not None:
        callbacks.append(EarlyStopping('lr-Adam',0,max_epochs,mode="min",check_finite=False,stopping_threshold=min_lr))
    #if trial is not None:
    #    callbacks.append(PyTorchLightningPruningCallback(trial,'validation_score'))
    #logger = logging.getLogger('lightning.pytorch.core')
    if not os.path.exists(root_dir + '/checkpoints/'):
        os.mkdir(root_dir + '/checkpoints/')
    #logger.addHandler(logging.FileHandler(root_dir + f'stdout.log'))
    csv_logger = CSVLogger(root_dir)
    version = csv_logger.version
    tb_logger = TensorBoardLogger(root_dir,version=version)
    trainer = Trainer(
        log_every_n_steps=100,
        default_root_dir=root_dir,
        enable_checkpointing=True,
        callbacks= callbacks,
        #logger=logger,
        max_epochs=max_epochs,
        logger=(csv_logger,tb_logger),
        # reload_dataloaders_every_n_epochs = 100
    )
    #trainer.early_stopping_callback = early_stop_callback # type: ignore
    if return_logger:
        return trainer, version, csv_logger
    return trainer, version