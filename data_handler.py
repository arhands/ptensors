from typing import Literal, Union
import lightning.pytorch as pl
import numpy as np
from torch import Tensor
from torch_geometric.datasets import ZINC, LRGBDataset, TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
# from transforms import PreprocessTransform_4 as PreprocessTransform
from torch_geometric.loader import DataLoader
from data import FancyDataObject, PtensObjects
from torch_geometric.data import InMemoryDataset

from sklearn.model_selection import StratifiedKFold

_tu_datasets = ['MUTAG','ENZYMES','PROTEINS','COLLAB','IMDB-BINARY','REDDIT_BINARY','IMDB-MULTI','NCI1','NCI109','PTC_MR']
tu_dataset_type = Literal['MUTAG','ENZYMES','PROTEINS','COLLAB','IMDB-BINARY','REDDIT_BINARY','IMDB-MULTI','NCI1','NCI109','PTC_MR']
dataset_type = Union[
    Literal['ZINC','ZINC-Full','ogbg-molhiv','peptides-struct','graphproperty','ogbg-moltox21'],tu_dataset_type
    ]
# def _get_path(base_path: str,name: dataset_type):
#     if name == 'ZINC':
#         return base_path + '/ZINC/'
#     else:
#         return base_path
# def _get_dataset(root: str, ds: dataset_type, transform, pretransform) -> ZINC|LRGBDataset|TUDataset|PygGraphPropPredDataset|GraphPropertyDataset:
#     if ds in ['ZINC','ZINC-Full']:
#         return ZINC(root,ds=='ZINC',transform=transform,pre_transform=pretransform)
#     elif ds in _tu_datasets:
#         return TUDataset(root,ds,)
# class DatasetCache:
#     def __init__(self, root: str) -> None:
#         self.root = root
#     def get_dataset(self, name: dataset_type, pre_transform: Transform, transform: Transform):
#         path = _get_path(self.root,name)
#         if os.path.exists(path):
#             return 
#         if not os.path.exists(_get_path(self.root + '/cache',name)):
            

class DataHandler(pl.LightningDataModule):
    splits: dict[Literal['train','val','test'],InMemoryDataset]
    batch_size: dict[Literal['train','val','test'],int]
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, transform) -> None:
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

        # self.num_folds : int = num_folds #type: ignore
        # self.seed : int = seed #type: ignore
        # self.pre_move_to_device = pre_move_to_device
    # def prepare_data(self) -> None:
    #     if self.ds_name == 'ZINC':
    #         for split in ['train','val','test']:
    #             ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
    #     elif self.ds_name in ['ogbg-molhiv','ogbg-moltox21']:
    #         PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform,transform=self.transform)
    #     elif self.ds_name == 'graphproperty':
    #         for split in ['train','val','test']:
    #             GraphPropertyDataset(self.root,split,pre_transform=self.pre_transform,transform=self.transform)
    #     elif self.ds_name == 'peptides-struct':
    #         for split in ['train','val','test']:
    #             LRGBDataset(self.root,self.ds_name,split,self.transform,self.pre_transform)
    #     elif self.ds_name in _tu_datasets:
    #         # for split in ['train','val','test']:
    #         TUDataset(self.root,self.ds_name,self.transform,self.pre_transform,use_edge_attr=True,use_node_attr=True)
    #     else:
    #         raise NotImplementedError(f'Dataset prepare for "{self.ds_name}" not implemented.')
        # return super().prepare_data()
    # def set_fold_idx(self, idx: int):
    #     if hasattr(self,'split_indices'):
    #         # adapted from GIN.
    #         train_idx, test_idx = self.split_indices[idx]
    #         self.splits = {
    #             'train' : self.ds[train_idx],
    #             'val' : self.ds[test_idx],
    #             'test' : self.ds[test_idx],
    #         }
    #     else:
    #         self.init_split_idx = idx
    def _get_dataloader(self, split: Literal['train','val','test'],shuffle: bool=False):
        return DataLoader(dataset=self.splits[split],batch_size=self.batch_size[split],shuffle=shuffle)

    def _get_splits(self) -> dict[Literal['train','val','test'],InMemoryDataset]:...

    def setup(self, stage: Literal['fit','test','predict']):#type: ignore
        self.splits = self._get_splits() 
        # if self.ds_name == 'ZINC':
        #     self.splits = {
        #         split : ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
        #         for split in ['train','val','test']
        #     }
        # elif self.ds_name in ['ogbg-molhiv','ogbg-moltox21']:
        #     ds = PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform,transform=self.transform)
        #     if self.ds_name == 'ogbg-molhiv':
        #         split_idx = ds.get_idx_split()
        #         self.splits = {
        #             'train' : ds[split_idx['train']],
        #             'val' : ds[split_idx['valid']],
        #             'test' : ds[split_idx['test']],
        #         }
        #     else:
        #         train_ind = round(0.8*len(ds))#we take the moleculenet splits. (copying https://github.com/rusty1s/himp-gnn/blob/master/train_tox21.py)
        #         val_ind = round(0.9*len(ds))
        #         self.splits = {
        #             'train' : ds[:train_ind],
        #             'val' : ds[train_ind:val_ind],
        #             'test' : ds[val_ind:],
        #         }
        # elif self.ds_name == 'graphproperty':
        #     self.splits = {
        #         split : GraphPropertyDataset(self.root,split,pre_transform=self.pre_transform,transform=self.transform)
        #         for split in ['train','val','test']
        #     }
        # elif self.ds_name == 'peptides-struct':
        #     self.splits = { 
        #         split : LRGBDataset(self.root,self.ds_name,split,self.transform,self.pre_transform) 
        #         for split in ['train','val','test']
        #     }
        # elif self.ds_name in _tu_datasets:
        #     # adapted from GIN
        #     ds = TUDataset(self.root,self.ds_name,self.transform,self.pre_transform,use_edge_attr=True,use_node_attr=True)
        #     skf = StratifiedKFold(self.num_folds,shuffle=True,random_state=self.seed)
        #     self.split_indices = list(skf.split(np.zeros(len(ds)),ds.y))
        #     self.ds = ds
        #     if hasattr(self,'init_split_idx'):
        #         self.set_fold_idx(self.init_split_idx)
        #         del self.init_split_idx
        # else:
        #     raise NotImplementedError(f'Dataset setup for "{self.ds_name}" not implemented.')
    
    def train_dataloader(self):
        return self._get_dataloader('train',True)
    def test_dataloader(self):
        return self._get_dataloader('test')
    def val_dataloader(self):
        return self._get_dataloader('val')
    # def on_after_batch_transfer(self, batch: FancyDataObject, dataloader_idx: int) -> tuple[Tensor, Tensor, Tensor,int, PtensObjects]:
    def on_after_batch_transfer(self, batch: FancyDataObject, dataloader_idx: int) -> tuple[Tensor, Tensor, Tensor, PtensObjects]:
        return batch.x, batch.edge_attr, batch.y, PtensObjects.from_fancy_data(batch)
        # return batch.x, batch.edge_attr, batch.y, batch.num_graphs, PtensObjects.from_fancy_data(batch)

#################################################################################################################################
# dataset specific data handlers
#################################################################################################################################

class ZINCDatasetHandler(DataHandler):
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, transform, subset: bool = True, num_folds=None, seed=0) -> None:
        super().__init__(root + '/ZINC', device, train_batch_size, val_batch_size, test_batch_size, pre_transform, transform)
        self.subset = subset
    
    def prepare_data(self) -> None:
        ZINC(self.root,self.subset,pre_transform=self.pre_transform,transform=self.transform)
        # for split in ['train','val','test']:
            # ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
    
    def _get_splits(self) -> dict[Literal['train', 'val', 'test'], InMemoryDataset]:
        return {#type: ignore
                split : ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
                for split in ['train','val','test']
            }

class OGBGDatasetHandler(DataHandler):
    ds_name: Literal['ogbg-molhiv','ogbg-moltox21']
    def __init__(self, root: str, ds_name: Literal['ogbg-molhiv','ogbg-moltox21'], device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, transform, subset: bool = True, num_folds=None, seed=0) -> None:
        super().__init__(root, device, train_batch_size, val_batch_size, test_batch_size, pre_transform, transform)
        self.ds_name = ds_name
    
    def prepare_data(self) -> None:
        PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform,transform=self.transform)
    
    def _get_splits(self) -> dict[Literal['train', 'val', 'test'], InMemoryDataset]:
        ds = PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform,transform=self.transform)
        splits: dict[Literal['train','val','test'], InMemoryDataset]
        if self.ds_name == 'ogbg-molhiv':
            split_idx = ds.get_idx_split()
            splits = {#type: ignore
                'train' : ds[split_idx['train']],
                'val' : ds[split_idx['valid']],
                'test' : ds[split_idx['test']],
            }
        else:
            train_ind = round(0.8*len(ds))#we take the moleculenet splits. (copying https://github.com/rusty1s/himp-gnn/blob/master/train_tox21.py)
            val_ind = round(0.9*len(ds))
            splits = {#type: ignore
                'train' : ds[:train_ind],
                'val' : ds[train_ind:val_ind],
                'test' : ds[val_ind:],
            }
        return splits

class TUDatasetHandler(DataHandler):
    ds_name: tu_dataset_type
    ds: TUDataset
    def __init__(self, root: str, ds_name: tu_dataset_type, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, transform, num_folds=None, seed=0) -> None:
        super().__init__(root, device, train_batch_size, val_batch_size, test_batch_size, pre_transform, transform)
        self.num_folds = num_folds
        self.seed = seed
        self.ds_name = ds_name
    
    def prepare_data(self) -> None:
        TUDataset(self.root,self.ds_name,self.transform,self.pre_transform,use_edge_attr=True,use_node_attr=True)
    def set_fold_idx(self, idx: int):
        self.split_idx = idx
        self.splits = self._get_splits()
        if hasattr(self,'split_indices'):
            # adapted from GIN.
            train_idx, test_idx = self.split_indices[idx]
            # self.train_idx = train_idx
            # self.test_idx = test_idx
            # self.splits = {
            #     'train' : self.ds[train_idx],
            #     'val' : self.ds[test_idx],
            #     'test' : self.ds[test_idx],
            # }
        # else:
        self.split_idx = idx
    def _get_splits(self) -> dict[Literal['train', 'val', 'test'], InMemoryDataset]:
        # adapted from GIN
        ds = TUDataset(self.root,self.ds_name,self.transform,self.pre_transform,use_edge_attr=True,use_node_attr=True)
        skf = StratifiedKFold(self.num_folds,shuffle=True,random_state=self.seed)#type: ignore
        if not hasattr(self,'split_indices'):
            self.split_indices = list(skf.split(np.zeros(len(ds)),ds.y))
        self.ds = ds
        # if hasattr(self,'init_split_idx'):
        #     train_idx, test_idx = self.split_indices[self.init_split_idx]
        #     del self.init_split_idx
        #     self.train_idx = t
        train_idx, test_idx = self.split_indices[self.split_idx]
        return {#type: ignore
                'train' : self.ds[train_idx],
                'val' : self.ds[test_idx],
                'test' : self.ds[test_idx],
            }