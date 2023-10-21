from __future__ import annotations
import torch
from torch.nn import Module, Parameter, Embedding
from torch import Tensor
from typing import Literal, Union, overload
from torch_scatter import scatter_sum
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder

class DummyEdgeEncoder(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
    def forward(self, edge_attr: Tensor) -> Tensor:
        return torch.zeros(edge_attr.size(0),self.hidden_dim,device=edge_attr.device)

def get_edge_encoder(hidden_dim: int,ds: Literal['ZINC','ogbg-molhiv','graphproperty','ogbg-moltox21','peptides-struct']) -> Union[BondEncoder,Embedding,DummyEdgeEncoder]:
    if ds == 'ZINC':
        return Embedding(4,hidden_dim)
    elif ds in ['ogbg-molhiv','ogbg-moltox21','peptides-struct']:
        return BondEncoder(hidden_dim)
    elif ds == 'graphproperty':
        return DummyEdgeEncoder(hidden_dim)
    else: 
        raise NameError(f'Dataset {ds} unknown.')

class GraphPropertyNodeEncoder(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = Embedding(2,hidden_dim-1)
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([
            self.emb(x[:,0].int()),
            x[:,1].unsqueeze(-1)
        ],-1)
def get_node_encoder(hidden_dim: int,ds: Literal['ZINC','ogbg-molhiv','graphproperty','peptides-struct','ogbg-moltox21']) -> Union[AtomEncoder,Embedding,GraphPropertyNodeEncoder]:
    if ds == 'ZINC':
        return Embedding(28,hidden_dim)
    elif ds in ['ogbg-molhiv','ogbg-moltox21','peptides-struct']:
        return AtomEncoder(hidden_dim)
    elif ds == 'graphproperty':
        return GraphPropertyNodeEncoder(hidden_dim)
    else: 
        raise NameError(f'Dataset {ds} unknown.')

# class AffineSum(Module):
#     def __init__(self, epsilon : float =0.5) -> None:
#         super().__init__()
#         self.epsilon = Parameter(torch.tensor(epsilon,requires_grad=True))
#     def forward(self, x: Tensor, y: Tensor):
#         return self.epsilon * x + (1 - self.epsilon) * y

class CycleEmbedding0(Module):
    def __init__(self, hidden_dim: int, ds: Literal['ZINC','ogbg-molhiv']) -> None:
        super().__init__()
        self.emb = get_node_encoder(hidden_dim,ds)
    def forward(self, x: Tensor, atom_to_cycle: Tensor):
        x = self.emb(x)
        return scatter_sum(x[atom_to_cycle[0]],atom_to_cycle[1],0)

class CycleEmbedding1(Module):
    def __init__(self, hidden_dim: int, ds: Literal['ZINC','ogbg-molhiv','peptides-struct','graphproperty','ogbg-moltox21']) -> None:
        super().__init__()
        self.emb = get_node_encoder(hidden_dim,ds)
        self.epsilon = Parameter(torch.tensor(0.,requires_grad=True))
    def forward(self, x: Tensor, atom_to_cycle: Tensor):
        x = self.emb(x)[atom_to_cycle[0]]
        return (1 + self.epsilon) * x + scatter_sum(x,atom_to_cycle[1],0)[atom_to_cycle[1]]