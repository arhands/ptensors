from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, EmbeddingBag, ModuleList
from torch import Tensor
from typing import List, Literal, NamedTuple, Union, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GINEConv, GINConv
from torch_scatter import scatter_sum
from objects import MultiScaleData_2, TransferData1
from ptensors0 import transfer0_0_bi_msg, transfer0_0_msg

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
    def forward(self, x: Tensor, atom_to_cycle: tuple[Tensor,Tensor]):
        x = self.emb(x)
        return scatter_sum(x[atom_to_cycle[0]],atom_to_cycle[1],0)

class SplitLayer(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True),
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True),
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge_index: Tensor) -> tuple[Tensor,Tensor]:
        node2edge_val = node_rep[node2edge_index[0]]
        node2edge_msg = torch.cat([node2edge_val,edge_rep[node2edge_index[1]]],-1)
        node2edge_msg = self.lvl_mlp_1(node2edge_msg)
        cat_edge_rep = scatter_sum(torch.cat([node2edge_val,node2edge_msg],-1),node2edge_index[1],0,dim_size=len(edge_rep))

        lvl_aggr_edge, lift_aggr = cat_edge_rep[:,node2edge_val.size(-1):], cat_edge_rep[:,:node2edge_val.size(-1)]
        
        lvl_aggr_edge = lvl_aggr_edge[node2edge_index[1]] # broadcasting back to node-edge pairs
        lvl_aggr_edge = lvl_aggr_edge - node2edge_msg # removing self-messages
        
        lvl_aggr = scatter_sum(lvl_aggr_edge,node2edge_index[0],0,dim_size=len(node_rep))

        node_out = self.lvl_mlp_2((1 + self.epsilon1) * node_rep + lvl_aggr)
        edge_out = self.lift_mlp((1 + self.epsilon2) * edge_rep + lift_aggr)

        return node_out, edge_out

class EdgeCycleSplitLayer(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True),
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(hidden_channels*2,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True),
            Linear(hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, edge_rep: Tensor, cycle_rep: Tensor, edge2cycle: TransferData1) -> tuple[Tensor,Tensor]:
        cycle2edge = edge2cycle.reverse()

        msg_dupe = [None]
        def msg_encoder(x: Tensor, y: Tensor) -> Tensor:
            msg = self.lvl_mlp_1(torch.cat([x,y],-1))
            msg_dupe[0] = msg
            return torch.cat([x,msg],-1)
        cat_cycle = transfer0_0_bi_msg(edge_rep,edge2cycle,msg_encoder,cycle_rep)
        lvl_aggr_cycle, lift_aggr = cat_cycle[:,edge_rep.size(1):], cat_cycle[:,:edge_rep.size(1)]
        lvl_aggr = transfer0_0_msg(lvl_aggr_cycle,cycle2edge,lambda x: x - msg_dupe[0])# we subtract out the outgoing messages.


        edge_out = self.lvl_mlp_2((1 + self.epsilon1) * edge_rep + lvl_aggr)

        cycle_out = self.lift_mlp((1 + self.epsilon2) * cycle_rep + lift_aggr)

        return edge_out, cycle_out

class ModelLayer(Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.node_edge = SplitLayer(hidden_channels)
        self.edge_cycle = EdgeCycleSplitLayer(hidden_channels)
        self.mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        
    def forward(self, node_rep: Tensor, edge_rep: Tensor, cycle_rep: Tensor, data: MultiScaleData_2, edge2cycle: TransferData1) -> tuple[Tensor,Tensor,Tensor]:
        node_out, edge_out_1 = self.node_edge(node_rep,edge_rep,data.node2edge_index)
        edge_out_2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,edge2cycle)

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
            Linear(hidden_dim,hidden_dim*2,False),
            BatchNorm1d(hidden_dim*2),
            ReLU(True),
        ) for _ in range(3)])
        self.lin = Linear(hidden_dim*2,1)
    def forward(self, data: MultiScaleData_2) -> Tensor:
        # initializing model
        node_rep = self.atom_encoder(data.x)
        # print(165,'node_rep.size()',node_rep.size())
        edge_rep = self.edge_encoder(data.edge_attr)
        cycle_rep = self.cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))

        edge2cycle = data.get_edge2cycle()

        # performing message passing
        for layer in self.layers:
            node_rep,edge_rep,cycle_rep = layer(node_rep,edge_rep,cycle_rep,data,edge2cycle)
            # print(174,'node_rep.size()',node_rep.size())
        # finalizing model
        # print(data.num_graphs,data.batch)
        reps = [
            global_add_pool(node_rep,data.batch,size=data.num_graphs),
            global_add_pool(edge_rep,data.edge_batch,size=data.num_graphs),
            global_add_pool(cycle_rep,torch.arange(data.num_graphs,device=cycle_rep.device).repeat_interleave(data.num_cycles),size=data.num_graphs)
            # global_add_pool(cycle_rep,data.cycle_batch,size=data.num_graphs)
        ]
        rep = torch.sum(torch.stack([mlp(rep) for mlp, rep in zip(self.pool_mlps,reps)]),0)
        return self.lin(rep)

"""
Epoch 547: 100%|████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:05<00:00,  7.29it/s, v_num=11, val_score=0.0913, train_loss=0.0606^

Testing DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 30.90it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_score         │    0.08064056932926178    │
└───────────────────────────┴───────────────────────────┘
Epoch 583:  76%|███████▌  | 60/79 [00:06<00:01,  9.79it/s, v_num=8, val_score=0.0869, train_loss=0.0539]         
"""