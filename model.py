from __future__ import annotations
import torch
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, Linear, Dropout, Parameter, Embedding, ModuleList
from torch import Tensor
from typing import Union, Optional, Tuple
import ptens as p
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from utils import ptensors_layer_type, domain_list_type, GraphMapCache, _scalar_mult, _sum

def get_torch_mlp(in_channels: int, dropout: float, bias: bool = True, out_channels: Optional[int] = None, hidden_channels : Optional[int] = None):
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
class MLP(Module):
    def __init__(self, in_channels: int, dropout: float, bias: bool = True, out_channels: Optional[int] = None, hidden_channels : Optional[int] = None) -> None:
        super().__init__()
        self.mlp = get_torch_mlp(in_channels, dropout, bias, out_channels, hidden_channels)
    def forward(self, x: ptensors_layer_type) -> ptensors_layer_type:
        res = self.mlp(x.torch())
        atoms = x.get_atoms()
        if isinstance(x,p.ptensors0):
            return p.ptensors0.from_matrix(res,atoms)
        elif isinstance(x,p.ptensors1):
            return p.ptensors1.from_matrix(res,atoms)
        return p.ptensors1.from_matrix(res,atoms)

class AtomEncoder(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = Embedding(22,hidden_dim)
    def forward(self, x: Tensor) -> p.ptensors0:
        x = self.emb(x.flatten())
        x = p.ptensors0.from_matrix(x)
        return x

class EdgeEncoder(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = Embedding(4,hidden_dim)
    
    def forward(self, edge_attr: Tensor, edges: domain_list_type) -> p.ptensors0:
        return p.ptensors0.from_matrix(self.emb(edge_attr),edges)

class GINE(Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.nn = MLP(hidden_dim,dropout)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.eps = Parameter(torch.tensor(0.),requires_grad=True)
    
    def forward(self, x: p.ptensors0, to_edges: p.graph, from_edges: p.graph, edge_attr: Tensor, edges: Tensor) -> p.ptensors0:
        x_edges : p.ptensors0 = x.gather(to_edges)
        x_edges = (x_edges + self.edge_encoder(edge_attr,edges)).relu(0.)

        x = _scalar_mult(x,self.eps + 1) + x_edges.gather(from_edges)

        x = self.nn(x)
        
        return x

class GIN_P1(Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.nn = get_torch_mlp(2*hidden_dim,out_channels=hidden_dim,dropout=dropout)
        self.lin = torch.nn.Linear(hidden_dim,2*hidden_dim)
        self.eps = Parameter(torch.tensor(0.),requires_grad=True)
    
    def forward(self, x: p.ptensors1, G: p.graph) -> p.ptensors0:
        atoms = x.get_atoms()
        x_transfer = x.transfer1(atoms,G)
        x_linmaps = x.linmaps1()
        
        x = torch.stack([
            self.lin(x.torch()),
            (1 + self.eps) * x_linmaps.torch(),
            x_transfer.torch(),
        ]).sum(0)

        x = self.nn(x)
        
        return p.ptensors1.from_matrix(x,atoms)

class ModelLayer(Module):
    def __init__(self, hidden_channels: int, dropout: float, residual: bool) -> None:
        super().__init__()
        self.gnn_P0 = GINE(hidden_channels,dropout)
        self.mlp_cycle_5_to_P0 = MLP(hidden_channels,dropout)
        self.mlp_cycle_6_to_P0 = MLP(hidden_channels,dropout)

        self.gnn_cycle_5 = GIN_P1(hidden_channels,dropout)
        self.mlp_P0_to_cycle_5 = MLP(hidden_channels,dropout)
        self.mlp_cycle_6_to_cycle_5 = MLP(in_channels=2*hidden_channels,out_channels=hidden_channels,dropout=dropout)
        
        self.gnn_cycle_6 = GIN_P1(hidden_channels,dropout)
        self.mlp_P0_to_cycle_6 = MLP(hidden_channels,dropout)
        self.mlp_cycle_5_to_cycle_6 = MLP(in_channels=2*hidden_channels,out_channels=hidden_channels,dropout=dropout)
        
        self.residual = residual
        
    def forward(self, edge_attr: Tensor, graphs: GraphMapCache, x_P0: p.ptensors0, x_cycle_5: Optional[p.ptensors1], x_cycle_6: Optional[p.ptensors1], edges: Tensor) -> Tuple[p.ptensors0,Optional[p.ptensors1],Optional[p.ptensors1]]:
        # creating lists of incoming representations
        P0_atoms = graphs.atoms['G']
        y_P0 = self.gnn_P0(x_P0,graphs['E','G'],graphs['G','E'],edge_attr,edges)

        if x_cycle_5 is not None:
            y_P0 = y_P0 + self.mlp_cycle_5_to_P0(x_cycle_5.transfer0(P0_atoms,graphs['G','C_5']))
            
            cycle_5_atoms = graphs.atoms['C_5']
            y_cycle_5 = self.gnn_cycle_5(x_cycle_5,graphs['C_5'])
            y_cycle_5 = y_cycle_5 + self.mlp_P0_to_cycle_5(x_P0.transfer1(cycle_5_atoms,graphs['C_5','G']))
        else:
            y_cycle_5 = None
        
        if x_cycle_6 is not None:
            cycle_6_atoms = graphs.atoms['C_6']
            y_P0 = y_P0 + self.mlp_cycle_6_to_P0(x_cycle_6.transfer0(P0_atoms,graphs['G','C_6']))

            y_cycle_6 = self.gnn_cycle_6(x_cycle_6,graphs['C_6'])
            y_cycle_6 = y_cycle_6 + self.mlp_P0_to_cycle_6(x_P0.transfer1(cycle_6_atoms,graphs['C_6','G']))

            if x_cycle_5 is not None:
                y_cycle_6 = y_cycle_6 + self.mlp_cycle_5_to_cycle_6(x_cycle_5.transfer1(cycle_6_atoms,graphs['C_6','C_5']))
                y_cycle_5 = y_cycle_5 + self.mlp_cycle_6_to_cycle_5(x_cycle_6.transfer1(cycle_5_atoms,graphs['C_5','C_6']))
        else:
            y_cycle_6 = None

        if self.residual:
            y_P0 = y_P0 + x_P0
            if x_cycle_5 is not None:
                y_cycle_5 = y_cycle_5 + x_cycle_5
            if x_cycle_6 is not None:
                y_cycle_6 = y_cycle_6 + x_cycle_6
        
        return y_P0, y_cycle_5, y_cycle_6


class Net(Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, residual: bool) -> None:
        super().__init__()
        # Initialization layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.cycle_5_mlp = MLP(hidden_dim,dropout)
        self.cycle_6_mlp = MLP(hidden_dim,dropout)

        # convolutional layers
        self.layers = ModuleList(ModelLayer(hidden_dim,dropout,residual) for _ in range(num_layers))
        # finalization layers
        self.final_mlp = Sequential(
            Linear(hidden_dim,2*hidden_dim),
            # Linear(3*hidden_dim,2*hidden_dim,False),
            BatchNorm1d(2*hidden_dim),
            ReLU(True),
            Linear(2*hidden_dim,1),
        )
    def forward(self, data: Data, graphs: GraphMapCache) -> Tensor:
        # initializing model
        x : p.ptensors0 = self.atom_encoder(data.x)
        edges = data.edge_index.transpose(1,0).tolist()
        edge_attr = data.edge_attr

        x_cycle_5 : Union[p.ptensors1,None] = self.cycle_5_mlp(x.unite1(graphs['C_5','G'])) if graphs['C_5'] is not None else None
        x_cycle_6 : Union[p.ptensors1,None] = self.cycle_6_mlp(x.unite1(graphs['C_6','G'])) if graphs['C_6'] is not None else None
        
        # performing message passing
        for layer in self.layers:
            x, x_cycle_5, x_cycle_6 = layer(edge_attr,graphs,x, x_cycle_5, x_cycle_6,edges)
        
        # finalizing model
        atoms = graphs.atoms['G']
        h = [x.torch()]
        if x_cycle_5 is not None:
            h.append(x_cycle_5.transfer0(atoms,graphs['G','C_5']).torch())
        if x_cycle_6 is not None:
            h.append(x_cycle_6.transfer0(atoms,graphs['G','C_6']).torch())
        
        # y : Tensor = torch.cat(h,-1)
        y : Tensor = _sum(h)
        y = global_mean_pool(y,data.batch)

        y = self.final_mlp(y)
        return y.flatten()
        



        
        