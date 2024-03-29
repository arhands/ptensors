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
from torch_geometric.transforms import BaseTransform, Compose
from data_transforms import StandardPreprocessing, encoding_flags, label_type

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
    
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform: BaseTransform, ltype: label_type, node_enc: encoding_flags, edge_enc: encoding_flags) -> None:
        super().__init__()
        # self.transform = PreprocessTransform()
        self.pre_transform = Compose(
            [
                StandardPreprocessing(ltype,node_enc,edge_enc),
                pre_transform
            ]
        )
        self.root = root
        self.device = device
        self.batch_size = {
            'train' : train_batch_size,
            'val' : val_batch_size,
            'test' : test_batch_size
        }
    def _get_dataloader(self, split: Literal['train','val','test'],shuffle: bool=False):
        return DataLoader(dataset=self.splits[split],batch_size=self.batch_size[split],shuffle=shuffle)

    def _get_splits(self) -> dict[Literal['train','val','test'],InMemoryDataset]:...

    def setup(self, stage: Literal['fit','test','predict']):#type: ignore
        self.splits = self._get_splits() 

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
    def __init__(self, root: str, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, subset: bool = True) -> None:
        super().__init__(root + '/ZINC', device, train_batch_size, val_batch_size, test_batch_size, pre_transform,'single-dim',None,None)
        self.subset: bool = subset

    def prepare_data(self) -> None:
        ZINC(self.root,self.subset,pre_transform=self.pre_transform)
        # for split in ['train','val','test']:
            # ZINC(self.root,self.subset,split,pre_transform=self.pre_transform,transform=self.transform)
    
    def _get_splits(self) -> dict[Literal['train', 'val', 'test'], InMemoryDataset]:
        return {#type: ignore
                split : ZINC(self.root,self.subset,split,pre_transform=self.pre_transform)
                for split in ['train','val','test']
            }

class OGBGDatasetHandler(DataHandler):
    ds_name: Literal['ogbg-molhiv','ogbg-moltox21']
    def __init__(self, root: str, ds_name: Literal['ogbg-molhiv','ogbg-moltox21'], device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform) -> None:
        if ds_name == 'ogbg-molhiv':
            ltype = 'single-dim'
        else:
            ltype = 'multi-label'
        super().__init__(root, device, train_batch_size, val_batch_size, test_batch_size, pre_transform,ltype,'OGB','OGB')
        self.ds_name = ds_name
    
    def prepare_data(self) -> None:
        PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform)
    
    def _get_splits(self) -> dict[Literal['train', 'val', 'test'], InMemoryDataset]:
        ds = PygGraphPropPredDataset(self.ds_name,self.root,pre_transform=self.pre_transform)
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
    def __init__(self, root: str, ds_name: tu_dataset_type, device: str, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, num_folds=None, seed=0) -> None:
        use_degree: encoding_flags = None if ds_name == 'REDDIT_BINARY' else "degree"
        if ds_name in ['COLLAB','IMDB-MULTI','ENZYMES']:
            ltype = 'multi-class'
        else:
            ltype = 'single-dim'
        super().__init__(root, device, train_batch_size, val_batch_size, test_batch_size, pre_transform,ltype,use_degree,use_degree)
        self.num_folds = num_folds
        self.seed = seed
        self.ds_name = ds_name
        self.split_idx = 0
    
    def prepare_data(self) -> None:
        TUDataset(self.root,self.ds_name,pre_transform=self.pre_transform,use_edge_attr=True,use_node_attr=True)
    def set_fold_idx(self, idx: int):
        self.split_idx = idx
        self.splits = self._get_splits()
    def _get_splits(self) -> dict[Literal['train', 'val', 'test'], InMemoryDataset]:
        # adapted from GIN
        ds = TUDataset(self.root,self.ds_name,pre_transform=self.pre_transform,use_edge_attr=True,use_node_attr=True)
        skf = StratifiedKFold(self.num_folds,shuffle=True,random_state=self.seed)#type: ignore
        if not hasattr(self,'split_indices'):
            self.split_indices = list(skf.split(np.zeros(len(ds)),ds.y))
        self.ds = ds
        train_idx, test_idx = self.split_indices[self.split_idx]
        return {#type: ignore
                'train' : self.ds[train_idx],
                'val' : self.ds[test_idx],
                'test' : self.ds[test_idx],
            }