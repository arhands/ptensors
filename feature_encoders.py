from __future__ import annotations
import torch
from torch.nn import Module, Parameter, Embedding
from torch import Tensor
from typing import Literal, Union
from torch_scatter import scatter_sum
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
def get_edge_encoder(hidden_dim: int,ds: Literal['ZINC','ogbg-molhiv']) -> Union[BondEncoder,Embedding]:
    if ds == 'ZINC':
        return Embedding(4,hidden_dim)
    elif ds == 'ogbg-molhiv':
        return BondEncoder(hidden_dim)
    else: 
        raise NameError(f'Dataset {ds} unknown.')

def get_node_encoder(hidden_dim: int,ds: Literal['ZINC','ogbg-molhiv']) -> Union[AtomEncoder,Embedding]:
    if ds == 'ZINC':
        return Embedding(22,hidden_dim)
    elif ds == 'ogbg-molhiv':
        return AtomEncoder(hidden_dim)
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
    def __init__(self, hidden_dim: int, ds: Literal['ZINC','ogbg-molhiv']) -> None:
        super().__init__()
        self.emb = get_node_encoder(hidden_dim,ds)
        self.epsilon = Parameter(torch.tensor(0.,requires_grad=True))
    def forward(self, x: Tensor, atom_to_cycle: Tensor):
        x = self.emb(x)[atom_to_cycle[0]]
        return (1 + self.epsilon) * x + scatter_sum(x,atom_to_cycle[1],0)[atom_to_cycle[1]]