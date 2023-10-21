import math
from typing import Literal
import lightning.pytorch as pl
from torch_geometric.datasets import ZINC, LRGBDataset
from ogb.graphproppred import PygGraphPropPredDataset
# from transforms import PreprocessTransform_4 as PreprocessTransform
from torch_geometric.loader import DataLoader
from graph_property import GraphPropertyDataset

class DataHandler(pl.LightningDataModule):
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, dataset: Literal['ZINC','ZINC-Full','ogbg-molhiv','peptides-struct','graphproperty','ogbg-moltox21'], pre_transform, transform) -> None:
        super().__init__()
        # self.transform = PreprocessTransform()
        self.pre_transform = pre_transform
        self.transform = transform
        self.root = root
        self.device = device
        self.batch_size = {
            'train' : train_batch_size,
            'val' : val_batch_size,
            'test' : test_batch_size
        }
        if dataset == 'ZINC-Full':
            self.subset = False
            dataset = 'ZINC'
        else:
            self.subset = True
        self.ds_name : Literal['ZINC','ogbg-molhiv','graphproperty','peptides-struct','ogbg-moltox21'] = dataset
        # self.pre_move_to_device = pre_move_to_device
    def prepare_data(self) -> None:
        if self.ds_name == 'ZINC':
            for split in ['train','val','test']:
                ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
        elif self.ds_name in ['ogbg-molhiv','ogbg-moltox21']:
            PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform,transform=self.transform)
        elif self.ds_name == 'graphproperty':
            for split in ['train','val','test']:
                GraphPropertyDataset(self.root,split,pre_transform=self.pre_transform,transform=self.transform)
        elif self.ds_name == 'peptides-struct':
            for split in ['train','val','test']:
                LRGBDataset(self.root,self.ds_name,split,self.transform,self.pre_transform)
        else:
            raise NotImplementedError(f'Dataset prepare for "{self.ds_name}" not implemented.')
        # return super().prepare_data()
    def _get_dataloader(self, split: Literal['train','val','test'],shuffle: bool=False):
        return DataLoader(self.splits[split],self.batch_size[split],shuffle,num_workers=4,pin_memory=True,pin_memory_device=self.device,prefetch_factor=3) #type: ignore
    def setup(self, stage: Literal['fit','test','predict']):
        if self.ds_name == 'ZINC':
            self.splits = {
                split : ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
                for split in ['train','val','test']
            }
        elif self.ds_name in ['ogbg-molhiv','ogbg-moltox21']:
            ds = PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform,transform=self.transform)
            if self.ds_name == 'ogbg-molhiv':
                split_idx = ds.get_idx_split()
                self.splits = {
                    'train' : ds[split_idx['train']],
                    'val' : ds[split_idx['valid']],
                    'test' : ds[split_idx['test']],
                }
            else:
                train_ind = round(0.8*len(ds))#we take the moleculenet splits. (copying https://github.com/rusty1s/himp-gnn/blob/master/train_tox21.py)
                val_ind = round(0.9*len(ds))
                self.splits = {
                    'train' : ds[:train_ind],
                    'val' : ds[train_ind:val_ind],
                    'test' : ds[val_ind:],
                }
        elif self.ds_name == 'graphproperty':
            self.splits = {
                split : GraphPropertyDataset(self.root,split,pre_transform=self.pre_transform,transform=self.transform)
                for split in ['train','val','test']
            }
        elif self.ds_name == 'peptides-struct':
            self.splits = { 
                split : LRGBDataset(self.root,self.ds_name,split,self.transform,self.pre_transform) 
                for split in ['train','val','test']
            }
        else:
            raise NotImplementedError(f'Dataset setup for "{self.ds_name}" not implemented.')
    
    def train_dataloader(self):
        return self._get_dataloader('train',True)
    def test_dataloader(self):
        return self._get_dataloader('test')
    def val_dataloader(self):
        return self._get_dataloader('val')