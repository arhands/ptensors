from __future__ import annotations
from data import PtensObjects
from torch_geometric.utils import segment 
# from torch.nn import Linear, BatchNorm1d, ReLU, Module
# from torch.nn.functional import dropout, relu

import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Parameter, ModuleList
# from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, EmbeddingBag, ModuleList
from torch import Tensor
# from torch_scatter import scatter_sum
from objects import TransferData1, atomspack1
from ptensors1 import linmaps1_1, transfer0_1, transfer1_0
from ptensors0 import transfer0_0
from torch.nn import functional as F
from data_handler import dataset_type

from feature_encoders import get_edge_encoder, get_node_encoder, CycleEmbedding1
from objects import TransferData0, TransferData1
# from objects2 import TransferData2

_inner_mlp_mult = 2

class SplitLayer0_0(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int, eps: float, momentum: float) -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge: TransferData0) -> tuple[Tensor,Tensor]:
        lift_aggr = transfer0_0(node_rep,node2edge)
        

        lvl_aggr_edge = self.lvl_mlp_1(torch.cat([lift_aggr,edge_rep],-1))

        # lvl_aggr_edge, lift_aggr = cat_edge_rep[:,:-node2edge_val.size(-1)], cat_edge_rep[:,-node2edge_val.size(-1):]
        
        lvl_aggr = transfer0_0(lvl_aggr_edge,node2edge.reverse())

        node_out = self.lvl_mlp_2((1 + self.epsilon1) * node_rep + lvl_aggr)
        edge_out = self.lift_mlp((1 + self.epsilon2) * edge_rep + lift_aggr)

        return node_out, edge_out

class SplitLayer0_1(Module):
    r"""
    Computes the lift layer for the higher rep and level layer for 
    """
    def __init__(self, hidden_channels: int, eps: float, momentum: float, reduce_ptensor: str, reduce_messages: str = 'sum') -> None:
        super().__init__()
        self.lift_mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.lvl_mlp_1 = Sequential(
            Linear(3*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.lvl_mlp_2 = Sequential(
            Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
            BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
            ReLU(True),
            Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.epsilon1_1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon1_2 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)

        self.reduce_messages = reduce_messages
        self.reduce_ptensor = reduce_ptensor
    def forward(self, node_rep: Tensor, edge_rep: Tensor, node2edge: TransferData1) -> tuple[Tensor,Tensor]:
        lift_aggr = transfer0_1(node_rep,node2edge,[self.reduce_ptensor,self.reduce_messages])
        

        lvl_aggr_edge = self.lvl_mlp_1(torch.cat([lift_aggr,edge_rep],-1))

        # lvl_aggr_edge, lift_aggr = cat_edge_rep[:,:-node2edge_val.size(-1)], cat_edge_rep[:,-node2edge_val.size(-1):]
        
        lvl_aggr = transfer1_0(lvl_aggr_edge,node2edge.reverse(),[self.reduce_ptensor,self.reduce_messages,self.reduce_ptensor],return_list=True)

        node_out = self.lvl_mlp_2((1 + self.epsilon1_1) * node_rep + (1 + self.epsilon1_2) * lvl_aggr[0] + lvl_aggr[1])
        edge_out = self.lift_mlp((1 + self.epsilon2) * linmaps1_1(edge_rep,node2edge.target,'mean') + lift_aggr)
        # edge_out = self.lift_mlp((1 + self.epsilon2) * linmaps1_1(edge_rep,node2edge.target,self.reduce_ptensor) + lift_aggr)

        return node_out, edge_out

class ModelLayer(Module):
    def __init__(self, hidden_channels: int, dropout: float,eps: float, momentum: float, reduce_ptensors: str) -> None:
        super().__init__()
        self.node_edge = SplitLayer0_0(hidden_channels, eps,momentum)
        self.edge_cycle = SplitLayer0_1(hidden_channels, eps, momentum, reduce_ptensors)
        self.mlp = Sequential(
            Linear(2*hidden_channels,hidden_channels,False),
            BatchNorm1d(hidden_channels,eps,momentum),
            ReLU(True)
        )
        self.dropout = dropout
        
    def forward(self, node_rep: Tensor, edge_rep: Tensor, cycle_rep: Tensor, data: PtensObjects) -> tuple[Tensor,Tensor,Tensor]:
        node_out, edge_out_1 = self.node_edge(node_rep,edge_rep,data[(('nodes','edges'),0)])
        edge_out_2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,data[(('edges','cycles'),1)])

        edge_out = self.mlp(torch.cat([edge_out_1,edge_out_2],-1))

        node_out = F.dropout(node_out,self.dropout,self.training)
        edge_out = F.dropout(edge_out,self.dropout,self.training)
        cycle_out = F.dropout(cycle_out,self.dropout,self.training)


        return node_out, edge_out, cycle_out

def get_out_dim(ds: dataset_type) -> int:
    multi = {
        'ogbg-moltox21'     : 12,
        'peptides-struct'   : 11,

        # tudatasets
        'ENZYMES'           : 6 ,
        'COLLAB'            : 3 ,
        'IMDB-MULTI'        : 3 ,

    }
    if ds in multi:
        return multi[ds]
    else:
        return 1

class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, dataset: dataset_type, readout: str, eps: float, momentum: float, reduce_ptensors: str) -> None:
        super().__init__()
        # assert dataset in ['ZINC','ogbg-molhiv','graphproperty','peptides-struct','ogbg-moltox21'], dataset
        # Initialization layers
        self.atom_encoder = get_node_encoder(hidden_dim,dataset)
        self.edge_encoder = get_edge_encoder(hidden_dim,dataset)
        self.cycle_encoder = CycleEmbedding1(hidden_dim,dataset)

        # convolutional layers
        self.layers = ModuleList(ModelLayer(hidden_dim,dropout,eps,momentum, reduce_ptensors) for _ in range(num_layers))
        self.readout = readout
        # finalization layers
        self.pool_mlps = ModuleList([Sequential(
            Linear(hidden_dim,hidden_dim*2,False),
            BatchNorm1d(hidden_dim*2,eps,momentum),
            ReLU(True),
        ) for _ in range(3)])
        self.lin = Linear(hidden_dim*2,get_out_dim(dataset))

        self.dropout = dropout
    def forward(self, x: Tensor, edge_attr: Tensor, data: PtensObjects) -> Tensor:
        # initializing model
        node_rep = self.atom_encoder(x)
        edge_rep = self.edge_encoder(edge_attr)
        cycle_rep = self.cycle_encoder(x,data[(('nodes','cycles'),1)])

        # performing message passing
        for layer in self.layers:
            node_rep,edge_rep,cycle_rep = layer(node_rep,edge_rep,cycle_rep,data)

        # finalizing model
        reps: list[Tensor] = []
        rep: Tensor
        name: str
        for rep, name in [(node_rep,'nodes'),(edge_rep,'edges'),(cycle_rep,'cycles')]:
            ap: atomspack1 = data[(name,0)]
            if isinstance(ap.raw_num_domains,int):
                reps.append(segment(rep,torch.tensor([len(rep)],dtype=torch.int32,device=rep.device),self.readout))
            else:
                A = torch.empty(len(ap.raw_num_domains) + 1,dtype=ap.raw_num_domains.dtype,device=ap.raw_num_domains.device)
                A[0] = 0
                A[1:] = ap.raw_num_domains.cumsum(0)
                reps.append(segment(rep,A,self.readout))
        rep = torch.sum(torch.stack([mlp(rep) for mlp, rep in zip(self.pool_mlps,reps)]),0)

        rep = F.dropout(rep,self.dropout,self.training)
        return self.lin(rep)
