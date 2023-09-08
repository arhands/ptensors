from typing import Literal
import lightning.pytorch as pl
from torch_geometric.datasets import ZINC
from transforms import PreprocessTransform
from torch_geometric.loader import DataLoader

class DataHandler(pl.LightningDataModule):
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int) -> None:
        super().__init__()
        self.transform = PreprocessTransform()
        self.root = root
        self.device = device
        self.batch_size = {
            'train' : train_batch_size,
            'val' : val_batch_size,
            'test' : test_batch_size
        }
    def prepare_data(self) -> None:
        for split in ['train','val','test']:
            ZINC(self.root,split,True,split,pre_transform=self.transform)
        # return super().prepare_data()
    def _get_dataset(self, split: Literal['train','val','test']):
        return DataLoader(self.splits[split],self.batch_size[split],num_workers=4,pin_memory=True,pin_memory_device=self.device,prefetch_factor=3)
    def setup(self, stage: Literal['fit','test','predict']):
        self.splits = {
            split : ZINC(self.root,True,split,pre_transform=PreprocessTransform())
            for split in ['train','val','test']
        }
    
    def train_dataloader(self):
        return self._get_dataset('train')
    def test_dataloader(self):
        return self._get_dataset('test')
    def val_dataloader(self):
        return self._get_dataset('val')