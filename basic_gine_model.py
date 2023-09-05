from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, ModuleList
from torch import Tensor
from typing import Literal, NamedTuple, Union, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter
from torch_geometric.nn import GINEConv
from objects import MultiScaleData

def get_mlp(in_channels: int, dropout: float, bias: bool = True, out_channels: Optional[int] = None, hidden_channels : Optional[int] = None):
    assert 0 <= dropout < 1, 'Dropout must fall in the interval [0,1).'
    if out_channels is None:
        out_channels = in_channels
    if hidden_channels is None:
        hidden_channels = 2 * out_channels
    layers = [
        Linear(in_channels,hidden_channels,False),
        BatchNorm1d(hidden_channels),
        ReLU(True),
        Linear(hidden_channels,hidden_channels,False),
        BatchNorm1d(hidden_channels),
        ReLU(True),
        Linear(hidden_channels,out_channels,bias),
    ]
    if dropout > 0:
        layers.append(Dropout(dropout))
    return Sequential(*layers)

def get_edge_encoder(hidden_dim: int,ds: Literal['ZINC']) -> Module:
    if ds == 'ZINC':
        return Embedding(4,hidden_dim)
    else: 
        raise NameError(f'Dataset {ds} unknown.')

def get_node_encoder(hidden_dim: int,ds: Literal['ZINC']) -> Module:
    if ds == 'ZINC':
        return Embedding(22,hidden_dim)
    else: 
        raise NameError(f'Dataset {ds} unknown.')

class ModelLayer(Module):
    def __init__(self, hidden_channels: int, dropout: float, residual: bool, dataset: Literal['ZINC']) -> None:
        super().__init__()
        # self.node_gnn = GINEConv(get_mlp(hidden_channels,dropout),train_eps=True)
        self.conv = GINEConv(get_mlp(hidden_channels,dropout),0,True)
        self.edge_encoder = get_edge_encoder(hidden_channels,dataset)

        self.residual = residual
        
    def forward(self, x: Tensor, data: MultiScaleData) -> tuple[Tensor,Tensor,Tensor]:
        print(x.size())
        print(data.edge_index.size())
        print(data.edge_attr.size())
        y = self.conv(x,data.edge_index,self.edge_encoder(data.edge_attr))

        if self.residual:
            y = y + x 

        return y


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: Literal['ZINC'], residual: bool) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)

        # convolutional layers
        self.layers = ModuleList([ModelLayer(hidden_dim,dropout,residual,dataset) for _ in range(num_layers)])
        # finalization layers
        self.final_mlp = Sequential(
            Linear(hidden_dim,2*hidden_dim),
            # Linear(3*hidden_dim,2*hidden_dim,False),
            BatchNorm1d(2*hidden_dim),
            ReLU(True),
            Linear(2*hidden_dim,1),
        )
    def forward(self, data: MultiScaleData) -> Tensor:
        # initializing model
        node_rep = self.atom_encoder(data.x)

        # performing message passing
        for layer in self.layers:
            node_rep = layer(node_rep,data)
        
        # finalizing model
        nodes = global_mean_pool(node_rep,data.batch,size=data.num_graphs)
        
        return self.final_mlp(nodes)