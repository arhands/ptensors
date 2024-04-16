from typing import Literal, Optional
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning import Trainer
def get_trainer(run_dir: str, max_epochs: int, min_lr: Optional[float], mode: Literal['min','max'], logger: WandbLogger, progress_bar: bool = True, show_model_summary: bool = True) -> Trainer:
    callbacks: list = [
        ModelCheckpoint(run_dir,monitor='val_score',mode=mode,save_top_k=1),
    ]
    if show_model_summary:
        callbacks.append(ModelSummary(4))
    if min_lr is not None:
        callbacks.append(EarlyStopping('lr-Adam',0,max_epochs,mode="min",check_finite=False,stopping_threshold=min_lr))
    trainer = Trainer(
        default_root_dir=run_dir,
        enable_checkpointing=True,
        callbacks= callbacks,
        max_epochs=max_epochs,
        logger=logger,
        enable_model_summary = True,
        enable_progress_bar=progress_bar
    )
    return trainer