from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, EmbeddingBag, ModuleList
from torch import Tensor
from typing import Literal, Union
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_sum
from objects import MultiScaleData_2, TransferData1
from ptensors1 import linmaps1_1, transfer0_1, transfer1_0
from torch.nn import functional as F

from feature_encoders import get_edge_encoder, get_node_encoder, CycleEmbedding1

_inner_mlp_mult = 2

# class SplitLayer0_0(Module):
#     r"""
#     Computes the lift layer for the higher rep and level layer for 
#     """
#     def __init__(self, hidden_channels: int) -> None:
#         super().__init__()
#         self.lift_mlp = Sequential(
#             Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
#             BatchNorm1d(hidden_channels*_inner_mlp_mult),
#             ReLU(True),
#             Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
#             BatchNorm1d(hidden_channels),
#             ReLU(True)
#         )
#         self.lvl_mlp_1 = Sequential(
#             Linear(2*hidden_channels,hidden_channels,False),
#             BatchNorm1d(hidden_channels),
#             ReLU(True)
#         )
#         self.lvl_mlp_2 = Sequential(
#             Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
#             BatchNorm1d(hidden_channels*_inner_mlp_mult),
#             ReLU(True),
#             Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
#             BatchNorm1d(hidden_channels),
#             ReLU(True)
#         )
#         self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
#         self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
#     def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge: TransferData0) -> tuple[Tensor,Tensor]:
#         lift_aggr = transfer0_0(node_rep,node2edge)
        

#         lvl_aggr_edge = self.lvl_mlp_1(torch.cat([lift_aggr,edge_rep],-1))

#         # lvl_aggr_edge, lift_aggr = cat_edge_rep[:,:-node2edge_val.size(-1)], cat_edge_rep[:,-node2edge_val.size(-1):]
        
#         lvl_aggr = transfer0_0(lvl_aggr_edge,node2edge.reverse())

#         node_out = self.lvl_mlp_2((1 + self.epsilon1) * node_rep + lvl_aggr)
#         edge_out = self.lift_mlp((1 + self.epsilon2) * edge_rep + lift_aggr)

#         return node_out, edge_out

class SplitLayer(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge_index: Tensor) -> tuple[Tensor,Tensor]:
        node2edge_val = node_rep[node2edge_index[0]]
        # node2edge_msg = torch.cat([node2edge_val,edge_rep[node2edge_index[1]]],-1)
        # node2edge_msg = self.lvl_mlp_1(node2edge_msg)
        # cat_edge_rep = scatter_sum(torch.cat([node2edge_msg,node2edge_val],-1),node2edge_index[1],0,dim_size=len(edge_rep))

        lift_aggr = scatter_sum(node2edge_val,node2edge_index[1],0,dim_size=len(edge_rep))
        lvl_aggr_edge = self.lvl_mlp_1(torch.cat([lift_aggr,edge_rep],-1))

        # lvl_aggr_edge, lift_aggr = cat_edge_rep[:,:-node2edge_val.size(-1)], cat_edge_rep[:,-node2edge_val.size(-1):]
        
        lvl_aggr_edge = lvl_aggr_edge[node2edge_index[1]] # broadcasting back to node-edge pairs
        # lvl_aggr_edge = lvl_aggr_edge - node2edge_msg # removing self-messages
        
        lvl_aggr = scatter_sum(lvl_aggr_edge,node2edge_index[0],0,dim_size=len(node_rep))

        node_out = self.lvl_mlp_2((1 + self.epsilon1) * node_rep + lvl_aggr)
        edge_out = self.lift_mlp((1 + self.epsilon2) * edge_rep + lift_aggr)

        return node_out, edge_out

class SplitLayer0_1(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(3*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.epsilon1_1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon1_2 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge: TransferData1) -> tuple[Tensor,Tensor]:
        lift_aggr = transfer0_1(node_rep,node2edge)
        

        lvl_aggr_edge = self.lvl_mlp_1(torch.cat([lift_aggr,edge_rep],-1))

        # lvl_aggr_edge, lift_aggr = cat_edge_rep[:,:-node2edge_val.size(-1)], cat_edge_rep[:,-node2edge_val.size(-1):]
        
        lvl_aggr = transfer1_0(lvl_aggr_edge,node2edge.reverse(),return_list=True)

        node_out = self.lvl_mlp_2((1 + self.epsilon1_1) * node_rep + (1 + self.epsilon1_2) * lvl_aggr[0] + lvl_aggr[1])
        edge_out = self.lift_mlp((1 + self.epsilon2) * linmaps1_1(edge_rep,node2edge.target,'mean') + lift_aggr)

        return node_out, edge_out


class ModelLayer(Module):
    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.node_edge = SplitLayer(hidden_channels)
        self.edge_cycle = SplitLayer0_1(hidden_channels)
        self.mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels),
            ReLU(True)
        )
        self.dropout = dropout
        
    def forward(self, node_rep: Tensor, edge_rep: Tensor, cycle_rep: Tensor, data: MultiScaleData_2, edge2cycle: TransferData1) -> tuple[Tensor,Tensor,Tensor]:
        node_out, edge_out_1 = self.node_edge(node_rep,edge_rep,data.node2edge_index)
        edge_out_2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,edge2cycle)

        edge_out = self.mlp(torch.cat([edge_out_1,edge_out_2],-1))

        node_out = F.dropout(node_out,self.dropout,self.training)
        edge_out = F.dropout(edge_out,self.dropout,self.training)
        cycle_out = F.dropout(cycle_out,self.dropout,self.training)
        return node_out, edge_out, cycle_out


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: Literal['ZINC','ogbg-molhiv'], residual: bool, readout: Literal['mean','sum']) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)
        self.edge_encoder = get_edge_encoder(hidden_dim,dataset)
        self.cycle_encoder = CycleEmbedding1(hidden_dim,dataset)

        # convolutional layers
        self.layers = ModuleList(ModelLayer(hidden_dim,dropout) for _ in range(num_layers))
        self.readout = readout
        # finalization layers
        self.pool_mlps = ModuleList([Sequential(
            Linear(hidden_dim,hidden_dim*2,False),
            BatchNorm1d(hidden_dim*2),
            ReLU(True),
        ) for _ in range(3)])
        self.lin = Linear(hidden_dim*2,1)
        self.dropout = dropout
    def forward(self, data: MultiScaleData_2) -> Tensor:
        # initializing model
        node_rep = self.atom_encoder(data.x)
        edge_rep = self.edge_encoder(data.edge_attr)
        cycle_rep = self.cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
        edge2cycle = data.get_edge2cycle()

        # performing message passing
        for layer in self.layers:
            node_rep,edge_rep,cycle_rep = layer(node_rep,edge_rep,cycle_rep,data,edge2cycle)
        
        # finalizing model
        pool_fn = global_add_pool if self.readout == 'sum' else global_mean_pool
        reps = [
            pool_fn(node_rep,data.batch,size=data.num_graphs),
            pool_fn(edge_rep,data.edge_batch,size=data.num_graphs),
            pool_fn(cycle_rep,data.cycle_batch,size=data.num_graphs)
        ]
        rep = torch.sum(torch.stack([mlp(rep) for mlp, rep in zip(self.pool_mlps,reps)]),0)
        
        rep = F.dropout(rep,self.dropout,self.training)
        return self.lin(rep)

"""
Epoch 547: 100%|████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:05<00:00,  7.29it/s, v_num=11, val_score=0.0913, train_loss=0.0606^


"""