from __future__ import annotations
import torch
from torch.nn import Module, Parameter, Embedding, EmbeddingBag, Linear, Sequential
from torch import Tensor
from typing import Literal, Union, overload
from torch_scatter import scatter_sum
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from data_handler import dataset_type
from objects1 import TransferData1
from ptensors1 import transfer0_1

class DummyEdgeEncoder(Module):
    r"""
    Equivalent to torch.nn.Embedding with default options.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        w = torch.normal(0,1,[1,hidden_dim])
        self.weight = torch.nn.Parameter(w,requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        return self.weight.broadcast_to(x.size(0),-1)


class DummyNodeEncoder(Module):
    r"""
    Equivalent to torch.nn.Embedding with default options.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # w = torch.normal(0,1,[1,hidden_dim])
        # self.weight = torch.nn.Parameter(w,requires_grad=True)
        self.weight = Embedding(1,hidden_dim)
    def forward(self, x: Tensor) -> Tensor:
        # return self.weight.broadcast_to(x.size(0),-1)
        return self.weight(torch.zeros_like(x,dtype=torch.int32))

class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == 1, x.size()
        x = x.flatten()
        return x
class ToInt(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.int()
def get_tu_node_encoder(deg_count: int, hidden_dim: int):
    return Embedding(deg_count,hidden_dim)
def get_tu_edge_encoder(deg_count: int, hidden_dim: int):
    return EmbeddingBag(deg_count,hidden_dim)
# TODO: move type-casting to preprocessing
def get_edge_encoder(hidden_dim: int,ds: dataset_type) -> Union[BondEncoder,Embedding,DummyEdgeEncoder]:
    return {
        'ZINC'              : Sequential(Flatten(),Embedding(4,hidden_dim))       ,
        'ZINC-Full'         : Embedding(4,hidden_dim)       ,
        
        # OGB/molecular
        'ogbg-molhiv'       : BondEncoder(hidden_dim)       ,
        'ogbg-moltox21'     : BondEncoder(hidden_dim)       ,
        'peptides-struct'   : BondEncoder(hidden_dim)       ,

        # synthetic
        'graphproperty'     : DummyEdgeEncoder(hidden_dim)  ,

        # TUDatasets
        'MUTAG'             : Embedding(4,hidden_dim)       ,
        'ENZYMES'           : get_tu_edge_encoder(10,hidden_dim)  ,
        'PROTEINS'          : get_tu_edge_encoder(26,hidden_dim)  ,
        #'COLLAB'            : DummyEdgeEncoder(hidden_dim)  ,
        'IMDB-MULTI'        : get_tu_edge_encoder(89,hidden_dim)  ,
        'IMDB-BINARY'       : get_tu_edge_encoder(136,hidden_dim)  ,
        'REDDIT-BINARY'     : get_tu_edge_encoder(1,hidden_dim)   ,
        'NCI1'              : get_tu_edge_encoder(5,hidden_dim)  ,
        'NCI109'            : get_tu_edge_encoder(6,hidden_dim)  ,
        'PTC_MR'            : Embedding(4,hidden_dim)       ,
    }[ds]

class GraphPropertyNodeEncoder(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = Embedding(2,hidden_dim-1)
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([
            self.emb(x[:,0].int()),
            x[:,1].unsqueeze(-1)
        ],-1)
def get_node_encoder(hidden_dim: int,ds: dataset_type) -> Union[AtomEncoder,Embedding,GraphPropertyNodeEncoder]:
    return {
        'ZINC'              : Sequential(Flatten(),Embedding(28,hidden_dim))              ,
        'ZINC-Full'         : Embedding(28,hidden_dim)              ,

        # OGB/Molecular
        'ogbg-molhiv'       : AtomEncoder(hidden_dim)               ,
        'ogbg-moltox21'     : AtomEncoder(hidden_dim)               ,
        'peptides-struct'   : AtomEncoder(hidden_dim)               ,

        # synthetic
        'graphproperty'     : GraphPropertyNodeEncoder(hidden_dim)  ,

        # TUDatasets
        'MUTAG'             : Embedding( 7,hidden_dim),
        'ENZYMES'           : Embedding( 3,hidden_dim)              ,
        'PROTEINS'          : Embedding( 3,hidden_dim)              ,
        # 'COLLAB'            : get_tu_node_encoder(hidden_dim)          ,
        'IMDB-MULTI'        : get_tu_node_encoder(89,hidden_dim)          ,
        'IMDB-BINARY'       : get_tu_node_encoder(136,hidden_dim)          ,
        'REDDIT-BINARY'     : get_tu_node_encoder(1,hidden_dim)       ,
        'NCI1'              : Embedding(37,hidden_dim)              ,
        'NCI109'            : Embedding(38,hidden_dim)              ,
        'PTC_MR'            : Embedding(18,hidden_dim)              ,
    }[ds]
    


# class AffineSum(Module):
#     def __init__(self, epsilon : float =0.5) -> None:
#         super().__init__()
#         self.epsilon = Parameter(torch.tensor(epsilon,requires_grad=True))
#     def forward(self, x: Tensor, y: Tensor):
#         return self.epsilon * x + (1 - self.epsilon) * y

class CycleEmbedding0(Module):
    def __init__(self, hidden_dim: int, ds: dataset_type) -> None:
        super().__init__()
        self.emb = get_node_encoder(hidden_dim,ds)
    def forward(self, x: Tensor, atom_to_cycle: Tensor):
        x = self.emb(x)
        return scatter_sum(x[atom_to_cycle[0]],atom_to_cycle[1],0)

class CycleEmbedding1(Module):
    def __init__(self, hidden_dim: int, ds: dataset_type) -> None:
        super().__init__()
        self.emb = get_node_encoder(hidden_dim,ds)
        self.epsilon = Parameter(torch.tensor(0.,requires_grad=True))
    def forward(self, x: Tensor, node2cycle: TransferData1) -> Tensor:
        x = self.emb(x)
        c = x.size(-1)
        x = transfer0_1(x,node2cycle)
        # x = (1 + self.epsilon) * x[:,c:] + x[:,:c]
        x = (1 + self.epsilon) * x[:,:c] + x[:,c:]
        return x