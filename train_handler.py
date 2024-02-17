import os
from typing import Any, Literal, Optional
# from typing import Any, Literal, Optional, overload
from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
# from pytorch_lightning.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, ModelSummary
from lightning import Trainer
# import wandb

# def get_dataset_mode(ds: Literal['ZINC','ogbg-molhiv']):
#     return {
#         'ZINC' : 'min',
#         'ogbg-molhiv' : 'max',
#     }[ds]
def get_trainer(root_dir: str, max_epochs: int, min_lr: Optional[float], mode: Literal['min','max'],wandb_project_name: Optional[str] = None, args: Optional[dict[str,Any]] = None,pos=0, progress_bar: bool = True, show_model_summary: bool = True) -> tuple[Trainer,int|str]:
    version: int|str
    if wandb_project_name is not None:
        assert args is not None
        from lightning.pytorch.loggers.wandb import WandbLogger
        logger = WandbLogger(wandb_project_name,root_dir)
        args = {k : args[k] for k in args if k not in ['device','eval_batch_size','wandb_project_name']}
        logger.log_hyperparams(args)
        version = logger.version#type: ignore
    else:
        logger = TensorBoardLogger(root_dir)
        version = logger.version
    bar = TQDMProgressBar(10 if progress_bar else 0,pos)
    callbacks = [
        ModelCheckpoint(root_dir + f'/checkpoints/version_{version}/',monitor='val_score',mode=mode,save_top_k=1),
        bar,
    ]
    if show_model_summary:
        callbacks.append(ModelSummary(4))
    if not os.path.exists(root_dir + '/checkpoints/'):
        os.mkdir(root_dir + '/checkpoints/')
    if not os.path.exists(root_dir + f'/checkpoints/version_{version}/'):
        os.mkdir(root_dir + f'/checkpoints/version_{version}/')
    if not os.path.exists(root_dir + '/lightning_logs/'):
        os.mkdir(root_dir + '/lightning_logs/')
    if min_lr is not None:
        callbacks.append(EarlyStopping('lr-Adam',0,max_epochs,mode="min",check_finite=False,stopping_threshold=min_lr))
    #if trial is not None:
    #    callbacks.append(PyTorchLightningPruningCallback(trial,'validation_score'))
    # import logging
    # console_logger = logging.getLogger('lightning.pytorch')
    # # console_logger = logging.getLogger('lightning.pytorch.core')
    # console_logger.addHandler(logging.FileHandler("core2.log"))
    # print("\n43\n")
    trainer = Trainer(
        default_root_dir=root_dir,
        enable_checkpointing=True,
        callbacks= callbacks,
        max_epochs=max_epochs,
        logger=logger,#type: ignore
        # **model_summary
        enable_model_summary = False,
        # enable_model_summary = show_model_summary
    )
    # print("\n54\n")
    return trainer, version