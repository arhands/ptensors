from __future__ import annotations
import torch
from torch.nn import Module, Parameter, Embedding, EmbeddingBag, Linear, Sequential
from torch import Tensor
from typing import Literal, Optional, Union, overload, cast
from torch_scatter import scatter_sum
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from data_handler import dataset_type
from objects1 import TransferData1
from ptensors1 import transfer0_1

class DummyNodeEncoder(Module):
    r"""
    Equivalent to torch.nn.Embedding with default options.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = Embedding(1,hidden_dim)
    def forward(self, x: Tensor) -> Tensor:
        return self.weight(torch.zeros_like(x,dtype=torch.int32))

def get_tu_node_encoder(deg_count: int, hidden_dim: int):
    return Embedding(deg_count,hidden_dim)
def get_tu_edge_encoder(deg_count: int, hidden_dim: int):
    return EmbeddingBag(deg_count,hidden_dim)
# TODO: move type-casting to preprocessing
@overload
def get_edge_encoder(hidden_dim: int, name: Literal['BondEncoder'], feature_count: Literal[None] = None) -> Module:...
@overload
def get_edge_encoder(hidden_dim: int, name: Literal['Embedding','EmbeddingBag'], feature_count: int) -> Module:...
def get_edge_encoder(hidden_dim: int, name: Literal['Embedding','BondEncoder','EmbeddingBag'], feature_count: Optional[int] = None) -> Module:
    if name == "BondEncoder":
        return BondEncoder(hidden_dim)
    else:
        feature_count = cast(int,feature_count)
        return {
            'Embedding' : lambda: Embedding(feature_count,hidden_dim),
            'EmbeddingBag' : lambda: EmbeddingBag(feature_count,hidden_dim),
        }[name]()


@overload
def get_node_encoder(hidden_dim: int, name: Literal['AtomEncoder'], num_features: Literal[None] = None) -> Module:...
@overload
def get_node_encoder(hidden_dim: int, name: Literal['Embedding'], num_features: int) -> Module:...
def get_node_encoder(hidden_dim: int, name: Literal['Embedding','AtomEncoder'], num_features: Optional[int] = None) -> Module:
    if name == 'AtomEncoder':
        return AtomEncoder(hidden_dim)
    else:
        num_features = cast(int,num_features)
        return Embedding(num_features,hidden_dim)

class CycleEmbedding(Module):
    def __init__(self, hidden_dim: int, node_emb: Module) -> None:
        super().__init__()
        self.emb = node_emb
        self.epsilon = Parameter(torch.tensor(0.,requires_grad=True))
    def forward(self, x: Tensor, node2cycle: TransferData1) -> Tensor:
        x = self.emb(x)
        c = x.size(-1)
        x = transfer0_1(x,node2cycle)
        # x = (1 + self.epsilon) * x[:,c:] + x[:,:c]
        x = (1 + self.epsilon) * x[:,:c] + x[:,c:]
        return x