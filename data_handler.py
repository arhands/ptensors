from typing import Literal
import lightning.pytorch as pl
from torch_geometric.datasets import ZINC
from ogb.graphproppred import PygGraphPropPredDataset
# from transforms import PreprocessTransform_4 as PreprocessTransform
from torch_geometric.loader import DataLoader

class DataHandler(pl.LightningDataModule):
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, dataset: Literal['ZINC','ogbg-molhiv'], pre_transform) -> None:
        super().__init__()
        # self.transform = PreprocessTransform()
        self.pre_transform = pre_transform
        self.root = root
        self.device = device
        self.batch_size = {
            'train' : train_batch_size,
            'val' : val_batch_size,
            'test' : test_batch_size
        }
        self.ds_name : Literal['ZINC','ogbg-molhiv'] = dataset
    def prepare_data(self) -> None:
        if self.ds_name == 'ZINC':
            for split in ['train','val','test']:
                ZINC(self.root,True,split,pre_transform=self.pre_transform)
        elif self.ds_name == 'ogbg-molhiv':
            PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform)
        else:
            raise NotImplementedError(f'Dataset prepare for "{self.ds_name}" not implemented.')
        # return super().prepare_data()
    def _get_dataloader(self, split: Literal['train','val','test'],shuffle: bool=False):
        return DataLoader(self.splits[split],self.batch_size[split],shuffle,num_workers=4,pin_memory=True,pin_memory_device=self.device,prefetch_factor=3) #type: ignore
    def setup(self, stage: Literal['fit','test','predict']):
        if self.ds_name == 'ZINC':
            self.splits = {
                split : ZINC(self.root,True,split,pre_transform=self.pre_transform)
                for split in ['train','val','test']
            }
        else:
            ds = PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform)
            split_idx = ds.get_idx_split()
            self.splits = {
                split : ds[split_idx[split]]
                for split in ['train','valid','test']
            }
    
    def train_dataloader(self):
        return self._get_dataloader('train',True)
    def test_dataloader(self):
        return self._get_dataloader('test')
    def val_dataloader(self):
        return self._get_dataloader('val')