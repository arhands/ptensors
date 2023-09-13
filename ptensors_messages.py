from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential, Parameter
from objects import TransferData1
from torch_scatter import scatter_sum

class Transfer0_1(Module):
    def forward(self, x: Tensor, y: Tensor)

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
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge_index: Tensor) -> Tuple[Tensor,Tensor]:
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