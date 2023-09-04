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

class Representation(NamedTuple):
    node_rep: Tensor
    edge_rep: Tensor
    cycle_rep: Tensor
    
    edge_attr: Tensor
    cycle_edge_attr: Tensor

    edge_index_node: Tensor
    edge_index_edge: Tensor
    edge_index_node_edge: Tensor
    edge_index_edge_cycle: Tensor

    cycle_edge_cycle_indicator: Tensor


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
    def __init__(self, hidden_channels: int, edge_encoder: Module, reduce: str = 'sum') -> None:
        super().__init__()
        self.reduce = reduce
        self.lin1 = Linear(hidden_channels,hidden_channels,bias=False)
        self.lin2 = Linear(hidden_channels,hidden_channels,bias=False)
        self.lin3 = Linear(hidden_channels,hidden_channels,bias=False)
        self.bn = BatchNorm1d(hidden_channels)
        self.mlp = get_mlp(hidden_channels,0)
        self.edge_encoder = edge_encoder
    def forward(self, node_rep: Tensor, edge_rep: Tensor, edge_attr: Tensor, edge_index: Tensor):
        print('node_rep.size()',node_rep.size())
        print('edge_rep.size()',edge_rep.size())
        print('edge_attr.size()',edge_attr.size())
        print('edge_index.size()',edge_index.size())
        print('edge_index.max()',edge_index.max())
        messages = (
            self.lin1(node_rep)[edge_index[0]] + 
            self.lin2(node_rep)[edge_index[1]] + 
            self.edge_encoder(edge_attr) + 
            self.lin3(edge_rep))
        messages = self.bn(messages).relu()
        y = scatter(messages,edge_index[1],0,reduce=self.reduce,dim_size=node_rep.size(0))
        return self.mlp(y)

class RaiseZero(Module):
    def __init__(self, hidden_channels: int, edge_encoder: Module, reduce: str = 'sum') -> None:
        super().__init__()
        self.reduce = reduce
        self.lin1 = Linear(hidden_channels,hidden_channels,bias=False)
        self.epsilon = Parameter(torch.tensor(0.),requires_grad=True)
        self.mlp1 = get_mlp(hidden_channels,0)
        self.mlp2 = get_mlp(hidden_channels,0)
        self.edge_encoder = edge_encoder
    def forward(self, edge_rep: Tensor, node_rep: Tensor, edge_index: Tensor):
        # edge_index: map from nodes to edges.
        messages = (
            self.lin1(node_rep)[edge_index[0]] + 
            (edge_rep * (1 + self.epsilon))[edge_index[1]])
        messages = self.mlp1(messages)
        y = scatter(messages,edge_index[1],0,reduce=self.reduce,dim_size=edge_rep.size(0))
        return self.mlp1(y)

class ModelLayer(Module):
    def __init__(self, hidden_channels: int, dropout: float, residual: bool, dataset: Literal['ZINC']) -> None:
        super().__init__()
        # self.node_gnn = GINEConv(get_mlp(hidden_channels,dropout),train_eps=True)
        self.node_gnn = ConvZero(hidden_channels,get_edge_encoder(hidden_channels,dataset))
        self.edge_gnn = ConvZero(hidden_channels,get_cycle_encoder(hidden_channels,dataset))
        self.node_edge_gnn = RaiseZero(hidden_channels,dataset)
        self.edge_cycle_gnn = RaiseZero(hidden_channels,dataset)

        self.lin1 = Linear(2*hidden_channels,hidden_channels)

        self.residual = residual
        
    def forward(self, rep: Representation) -> Representation:
        node_out = self.node_gnn(rep.node_rep,rep.edge_rep,rep.edge_attr,rep.edge_index_node)
        edge_out_1 = self.edge_gnn(rep.edge_rep,rep.cycle_rep[rep.cycle_edge_cycle_indicator],rep.edge_index_node_edge,rep.edge_index_edge)
        edge_out_2 = self.node_edge_gnn(rep.edge_rep,rep.node_rep,rep.edge_index_node_edge)
        edge_out = self.lin1(torch.cat([edge_out_1,edge_out_2],-1))
        
        cycle_out = self.edge_cycle_gnn(rep.cycle_rep,rep.edge_rep,rep.edge_index_edge_cycle)

        if self.residual:
            node_out = node_out + rep.node_rep
            edge_out = edge_out + rep.edge_rep
            cycle_out = cycle_out + rep.cycle_rep

        return Representation(
            node_out,
            edge_out,
            cycle_out,
            rep.edge_attr,
            rep.cycle_edge_attr,
            rep.edge_index_node,
            rep.edge_index_edge,
            rep.edge_index_node_edge,
            rep.edge_index_edge_cycle,
            rep.cycle_edge_cycle_indicator,
        )


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: Literal['ZINC'], residual: bool) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)
        self.edge_encoder = get_edge_encoder(hidden_dim,dataset)
        self.cycle_encoder = get_cycle_encoder(hidden_dim,dataset)

        # convolutional layers
        self.layers = ModuleList(ModelLayer(hidden_dim,dropout,residual,dataset) for _ in range(num_layers))
        # finalization layers
        self.final_mlp = Sequential(
            Linear(hidden_dim*3,2*hidden_dim),
            # Linear(3*hidden_dim,2*hidden_dim,False),
            BatchNorm1d(2*hidden_dim),
            ReLU(True),
            Linear(2*hidden_dim,1),
        )
    def forward(self, data: MultiScaleData) -> Tensor:
        # initializing model
        rep = Representation(
            self.atom_encoder(data.x),
            self.edge_encoder(data.edge_attr),
            self.cycle_encoder(data.edge_attr_cycle),
            data.edge_attr,
            data.edge_attr_cycle,
            data.edge_index,
            data.edge_index_edge,
            data.edge_index_node_edge,
            data.edge_index_edge_cycle,
            data.cycle_edge_cycle_indicator,
        )

        # performing message passing
        for layer in self.layers:
            rep = layer(rep)
        
        # finalizing model
        nodes, edges, cycles = rep.node_rep, rep.edge_rep, rep.cycle_rep
        nodes = global_mean_pool(nodes,data.batch,size=data.num_graphs)
        edges = global_mean_pool(edges,data.edge_batch,size=data.num_graphs)
        cycles = global_mean_pool(cycles,data.cycle_batch,size=data.num_graphs)

        return self.final_mlp(torch.cat([nodes,edges,cycles],-1))