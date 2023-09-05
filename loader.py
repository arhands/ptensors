from typing import Literal
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from transforms import PreprocessTransform

def get_dataloader(root: str, split: Literal['train','val','test'], batch_size: int, pin_memory_device: str) -> DataLoader:
    return DataLoader(ZINC(root,True,split,pre_transform=PreprocessTransform()),batch_size,split==split,num_workers=4,pin_memory=True,pin_memory_device=pin_memory_device,prefetch_factor=3)