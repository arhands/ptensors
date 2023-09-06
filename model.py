from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, ModuleList
from torch import Tensor
from typing import Literal, NamedTuple, Union, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from ptensors0 import TransferData0, transfer0_0
from torch_scatter import scatter
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

class ConvZero(Module):
    def __init__(self, hidden_channels: int, bias: bool, reduce: str) -> None:
        super().__init__()
        self.reduce = reduce
        self.lin1 = Linear(hidden_channels,2*hidden_channels,bias=False)
        self.lin2 = Linear(hidden_channels,2*hidden_channels,bias=False)
        self.lin3 = Linear(hidden_channels,2*hidden_channels,bias=False)
        self.mlp1 = Sequential(
            BatchNorm1d(2*hidden_channels),
            ReLU(True),
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True),
        )
        self.mlp2 = get_mlp(hidden_channels,0,bias)
    def forward(self, edge_rep: Tensor, face_rep: Tensor, edge_index: Tensor, face_to_message_indicator: Tensor):
        messages = (
            self.lin1(edge_rep)[edge_index[0]] + 
            self.lin2(edge_rep)[edge_index[1]] + 
            self.lin3(face_rep)[face_to_message_indicator]
        )
        messages = self.mlp1(messages)
        y = scatter(messages,edge_index[1],0,reduce=self.reduce,dim_size=edge_rep.size(0))
        return self.mlp2(y)

class RaiseZero(Module):
    def __init__(self, hidden_channels: int, edge_encoder: Module,bias: bool, reduce: str) -> None:
        super().__init__()
        self.reduce = reduce
        self.lin1 = Linear(hidden_channels,hidden_channels,bias=False)
        self.epsilon = Parameter(torch.tensor(0.),requires_grad=True)
        self.mlp1 = get_mlp(hidden_channels,0,final_linear=False)
        self.mlp2 = get_mlp(hidden_channels,0,bias,bias)
        self.edge_encoder = edge_encoder
    def forward(self, face_rep: Tensor, edge_rep: Tensor, domain_indicator: Tensor):
        # edge_index: map from edges to faces (i.e., lower dim reps to higher dim reps).
        messages = self.mlp1(torch.cat([face_rep[domain_indicator],edge_rep],-1))
        aggregates = scatter(messages,domain_indicator,0,reduce=self.reduce,dim_size=face_rep.size(0))
        return self.mlp2(aggregates)

class ModelLayer(Module):
    def __init__(self, hidden_channels: int, dropout: float, residual: bool, dataset: Literal['ZINC'], bias: bool) -> None:
        super().__init__()
        self.node_gnn = ConvZero(hidden_channels,bias,'sum')
        self.edge_gnn = ConvZero(hidden_channels,bias,'mean')
        self.node_edge_gnn = RaiseZero(hidden_channels,dataset,True,'sum')
        self.edge_cycle_gnn = RaiseZero(hidden_channels,dataset,True,'mean')

        self.edge_encoder = get_edge_encoder(hidden_channels,dataset)
        self.cycle_encoder = get_cycle_encoder(hidden_channels,dataset)

        self.residual = residual
        
    def forward(self, node_rep: Tensor, edge_rep: Tensor, cycle_rep: Tensor, data: MultiScaleData) -> tuple[Tensor,Tensor,Tensor]:
        node_out = self.node_gnn(node_rep,edge_rep,data.edge_index,data.node_edge_message_indicator)

        edge_out_1 = self.edge_gnn(edge_rep,cycle_rep[data.cycle2edge_msg_ind],data.edge_attr_cycle_edge,data.edge_index_edge)
        edge_out_2 = self.node_edge_gnn(edge_rep,node_rep,data.node_to_edge_indicator)
        edge_out = edge_out_1 + edge_out_2 + self.edge_encoder(data.edge_attr)
        
        cycle_out = self.edge_cycle_gnn(cycle_rep,edge_rep,data.edge_to_cycle_indicator) + self.cycle_encoder(data.cycle_attr)

        if self.residual:
            node_out = node_out + node_rep
            edge_out = edge_out + edge_rep
            cycle_out = cycle_out + cycle_rep

        return node_out, edge_out, cycle_out


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: Literal['ZINC'], residual: bool) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)
        self.edge_encoder = get_edge_encoder(hidden_dim,dataset)
        self.cycle_encoder = get_cycle_encoder(hidden_dim,dataset)

        # convolutional layers
        self.layers = ModuleList(ModelLayer(hidden_dim,dropout,residual,dataset,i+1==num_layers) for i in range(num_layers))
        # finalization layers
        self.final_mlp = Sequential(
            BatchNorm1d(hidden_dim),
            ReLU(True),
            Linear(hidden_dim,1),
        )
    def forward(self, data: MultiScaleData) -> Tensor:
        # initializing model
        node_rep = self.atom_encoder(data.x)
        edge_rep = self.edge_encoder(data.edge_attr)
        cycle_rep = self.cycle_encoder(data.cycle_attr)

        # performing message passing
        for layer in self.layers:
            node_rep,edge_rep,cycle_rep = layer(node_rep,edge_rep,cycle_rep,data)
        
        # finalizing model
        nodes = global_mean_pool(node_rep,data.batch,size=data.num_graphs)
        edges = global_mean_pool(edge_rep,data.edge_batch,size=data.num_graphs)
        cycles = global_mean_pool(cycle_rep,data.cycle_batch,size=data.num_graphs)
        
        return self.final_mlp(nodes + edges + cycles)