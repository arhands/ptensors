from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, EmbeddingBag, ModuleList
from torch import Tensor
from typing import List, Literal, NamedTuple, Union, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GINEConv, GINConv
from torch_scatter import scatter_sum
from objects import MultiScaleData

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

class CycleEmbedding(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = Embedding(22,hidden_dim)
    def forward(self, x: Tensor, cycle_ind: Tensor):
        x = self.emb(x)
        return scatter_sum(x,cycle_ind,0)
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
    def forward(self, node_rep: Tensor, node2edge_index: Tensor, edge_rep: Tensor) -> Tensor:
        # We first scatter from vertices to vertex x edge pairs.
        # We then use the first mlp and then sum across the edges.
        # Lastly, we map these edge-aggregate values to their respective members.
        messages = torch.cat([node_rep[node2edge_index[0]],edge_rep[node2edge_index[1]]],-1)
        messages = self.mlp1(messages)
        edge_rep = scatter_sum(messages,node2edge_index[1],0,dim_size=len(edge_rep))
        aggregate = scatter_sum(edge_rep[node2edge_index[1]],node2edge_index[0],0,dim_size=len(node_rep))
        return self.mlp2((1 + self.epsilon) * node_rep + aggregate)
class LiftLayer(Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(hidden_channels,2*hidden_channels,False),
            BatchNorm1d(2*hidden_channels),
            ReLU(True),
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_index: Union[Tensor,List[Tensor]], edge_rep: Tensor) -> Tensor:
        if len(edge_index) > 1:
            x = scatter_sum(node_rep[edge_index[0]],edge_index[1],0,dim_size=edge_rep.size(0))
        ident = (1 + self.epsilon)*edge_rep
        # print(node_rep.size(),edge_rep.size(),x.size(),edge_index.size(),ident.size())
        raw = x + ident
        return self.mlp(raw)

class SplitLayer(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(hidden_channels,2*hidden_channels,False),
            BatchNorm1d(2*hidden_channels),
            ReLU(True),
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,2*hidden_channels,False),
            BatchNorm1d(2*hidden_channels),
            ReLU(True),
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge_index: Tensor) -> tuple[Tensor,Tensor]:
        node2edge_val = node_rep[node2edge_index[0]]
        node2edge_msg = torch.cat([node2edge_val,edge_rep[node2edge_index[1]]],-1)
        node2edge_msg = self.lvl_mlp_1(node2edge_msg)
        cat_edge_rep = scatter_sum(torch.cat([node2edge_msg,node2edge_val],-1),node2edge_index[1],0,dim_size=len(edge_rep))

        lvl_aggr_edge, lift_aggr = cat_edge_rep[:,:-node2edge_val.size(-1)], cat_edge_rep[:,-node2edge_val.size(-1):]
        
        lvl_aggr_edge = lvl_aggr_edge[node2edge_index[1]] # broadcasting back to node-edge pairs
        lvl_aggr_edge = lvl_aggr_edge - node2edge_msg # removing self-messages
        
        lvl_aggr = scatter_sum(lvl_aggr_edge,node2edge_index[0],0,dim_size=len(node_rep))

        node_out = self.lvl_mlp_2((1 + self.epsilon1) * lvl_aggr)
        edge_out = self.lift_mlp((1 + self.epsilon2) * edge_rep + lift_aggr)

        return node_out, edge_out
class ModelLayer(Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.node_edge = SplitLayer(hidden_channels)
        self.edge_cycle = SplitLayer(hidden_channels)
        self.mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        
    def forward(self, node_rep: Tensor, edge_rep: Tensor, cycle_rep: Tensor, data: MultiScaleData) -> tuple[Tensor,Tensor,Tensor]:
        node_out, edge_out_1 = self.node_edge(node_rep,edge_rep,data.node2edge_index)
        edge_out_2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,data.edge2cycle_index)

        edge_out = self.mlp(torch.cat([edge_out_1,edge_out_2],-1))
        
        return node_out, edge_out, cycle_out


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: Literal['ZINC'], residual: bool) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)
        self.edge_encoder = get_edge_encoder(hidden_dim,dataset)
        self.cycle_encoder = CycleEmbedding(hidden_dim)

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
        cycle_rep = self.cycle_encoder(data.cycle_attr,data.cycle_ind)

        # performing message passing
        for layer in self.layers:
            node_rep,edge_rep,cycle_rep = layer(node_rep,edge_rep,cycle_rep,data)
        
        # finalizing model
        reps = [
            global_add_pool(node_rep,data.batch,size=data.num_graphs),
            global_add_pool(edge_rep,data.edge_batch,size=data.num_graphs),
            global_add_pool(cycle_rep,data.cycle_batch,size=data.num_graphs)
        ]
        rep = torch.sum(torch.stack([mlp(rep) for mlp, rep in zip(self.pool_mlps,reps)]),0)
        return self.lin(rep)