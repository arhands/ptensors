from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, ModuleList
from torch import Tensor
from typing import Literal, NamedTuple, Union, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GINEConv, GINConv
from torch_scatter import scatter_sum
from objects import MultiScaleData

def get_mlp(in_channels: int, dropout: float, bias: bool = True, final_linear: bool = True, out_channels: Optional[int] = None, hidden_channels : Optional[int] = None):
    assert 0 <= dropout < 1, 'Dropout must fall in the interval [0,1).'
    if out_channels is None:
        out_channels = in_channels
    if hidden_channels is None:
        hidden_channels = 2 * out_channels
    layers = [
        Linear(in_channels,hidden_channels,False),
        BatchNorm1d(hidden_channels),
        ReLU(True),
    ]
    if final_linear:
        layers.extend([
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True),
            Linear(hidden_channels,out_channels,bias),
        ])
    else:
        layers.extend([
            Linear(hidden_channels,out_channels,False),
            BatchNorm1d(out_channels),
            ReLU(True),
        ])
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

def get_cycle_encoder(hidden_dim: int,ds: Literal['ZINC']) -> Module:
    if ds == 'ZINC':
        return Embedding(19,hidden_dim)
    else: 
        raise NameError(f'Dataset {ds} unknown.')

class LevelConv(Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.mlp1 = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.mlp2 = Sequential(
            Linear(hidden_channels,2*hidden_channels,False),
            BatchNorm1d(2*hidden_channels),
            ReLU(True),
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_index: Tensor, edge_rep: Tensor) -> Tensor:
        messages = torch.cat([node_rep[edge_index[0]],edge_rep],-1)
        messages = self.mlp1(messages)
        aggregates = scatter_sum(messages,edge_index[1],0,dim_size=node_rep.size(0))
        return self.mlp2((1 + self.epsilon) * node_rep + aggregates)
class LiftLayer(Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_index: Tensor, edge_rep: Tensor) -> Tensor:
        agg = scatter_sum(node_rep[edge_index[0]],edge_index[1],0,dim_size=edge_rep.size(1))
        ident = (1 + self.epsilon)*edge_rep
        print(agg.size(),ident.size())
        raw = agg + ident
        return self.mlp(raw)

class ModelLayer(Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lvl_node = LevelConv(hidden_channels)
        self.lvl_edge = LevelConv(hidden_channels)
        self.lvl_cycle = LevelConv(hidden_channels)
        self.lft_edge = LiftLayer(hidden_channels)
        self.lft_cycle = LiftLayer(hidden_channels)
        self.mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        
    def forward(self, node_rep: Tensor, edge_rep: Tensor, cycle_rep: Tensor, data: MultiScaleData) -> tuple[Tensor,Tensor,Tensor]:
        node_out = self.lvl_node(node_rep,data.edge_index,edge_rep[data.edge2node_msg_ind])

        edge_out_1 = self.lvl_edge(edge_rep,data.edge2edge_edge_index,cycle_rep[data.cycle2edge_msg_ind])
        edge_out_2 = self.lft_edge(node_rep,[data.edge_index[0],data.edge2node_msg_ind],edge_rep)
        edge_out = self.mlp(torch.cat([edge_out_1,edge_out_2],-1))
        
        cycle_out = self.lft_cycle(edge_rep,data.edge2cycle_edge_index,cycle_rep)

        return node_out, edge_out, cycle_out


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: Literal['ZINC'], residual: bool) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)
        self.edge_encoder = get_edge_encoder(hidden_dim,dataset)
        self.cycle_encoder = get_cycle_encoder(hidden_dim,dataset)

        # convolutional layers
        self.layers = ModuleList(ModelLayer(hidden_dim) for _ in range(num_layers))
        # finalization layers
        self.pool_mlps = ModuleList([Sequential(
            Linear(hidden_dim,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU(True),
        ) for _ in range(3)])
        self.lin = Linear(hidden_dim,1)
    def forward(self, data: MultiScaleData) -> Tensor:
        # initializing model
        node_rep = self.atom_encoder(data.x)
        edge_rep = self.edge_encoder(data.edge_attr)
        cycle_rep = self.cycle_encoder(data.cycle_attr)

        # performing message passing
        for layer in self.layers:
            node_rep,edge_rep,cycle_rep = layer(node_rep,edge_rep,cycle_rep,data)
        
        # finalizing model
        reps = [
            global_add_pool(node_rep,data.batch,size=data.num_graphs),
            global_add_pool(edge_rep,data.edge_batch,size=data.num_graphs),
            global_add_pool(cycle_rep,data.cycle_batch,size=data.num_graphs)
        ]
        rep = torch.sum([mlp(rep) for mlp, rep in zip(self.pool_mlps,reps)])
        return self.lin(rep)