import os
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning import Trainer

def get_trainer(root_dir: str, max_epochs: int, min_lr: float) -> Trainer:
    mode = 'min'
    callbacks = [
        ModelCheckpoint(root_dir + '/checkpoints/',monitor='val_score',mode=mode,save_top_k=1),
        ModelSummary(4),
        TQDMProgressBar(refresh_rate=100),
        LearningRateMonitor("epoch"),
        EarlyStopping('lr',0,max_epochs,mode="min",check_finite=False,stopping_threshold=min_lr)
    ]
    #if trial is not None:
    #    callbacks.append(PyTorchLightningPruningCallback(trial,'validation_score'))
    #logger = logging.getLogger('lightning.pytorch.core')
    if not os.path.exists(root_dir + '/checkpoints/'):
        os.mkdir(root_dir + '/checkpoints/')
    #logger.addHandler(logging.FileHandler(root_dir + f'stdout.log'))
    trainer = Trainer(
        log_every_n_steps=10,
        default_root_dir=root_dir,
        enable_checkpointing=True,
        callbacks= callbacks,
        #logger=logger,
        max_epochs=max_epochs,
        logger=CSVLogger(root_dir),
        # reload_dataloaders_every_n_epochs = 100
    )
    #trainer.early_stopping_callback = early_stop_callback # type: ignore
    return trainer