from __future__ import annotations
from typing import Union
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Data
from objects import TransferData1, atomspack, atomspack2

def linmaps0_1(x: Tensor, domains:  atomspack):
    return x[domains.domain_indicator]

def linmaps1_0(x: Tensor, domains: atomspack, reduce: str='sum'):
    return scatter(x,domains.domain_indicator,0,reduce=reduce)

def linmaps1_1(x: Tensor, domains: atomspack, reduce: str='sum'):
    return torch.cat([
        x,
        linmaps0_1(linmaps1_0(x,domains,reduce),domains),
    ],-1)

def transfer0_1(x: Tensor, data: TransferData1, reduce: Union[list[str],str]='sum'):
    r"""
        for transfering from a ptensors0 to a ptensors1
        first reduce: reductions between different ptensors on specific atoms.
        second reduce: reductions between different ptensors on for incoming values (entire ptensors).
    """
    if isinstance(reduce,str):
        reduce = [reduce]*2
    domain_reduced = scatter(x[data.source.domain_indicator],data.source.atoms,0,dim_size=data.num_nodes,reduce=reduce[0])

    target_broadcasted = domain_reduced[data.target.atoms]

    intersection_broadcasted = scatter(x[data.domain_map_edge_index[0]],data.domain_map_edge_index[1],0,dim_size=data.num_targets,reduce=reduce[1])[data.target.domain_indicator]
    return torch.cat([
        target_broadcasted,
        intersection_broadcasted,
    ],-1)

def transfer1_0(x: Tensor, data: TransferData1, reduce: Union[list[str],str]='sum'):
    r"""for transfering from a ptensors0 to a ptensors1"""
    if isinstance(reduce,str):
        reduce = [reduce]*4
    
    domain_reduced = scatter(x,data.source.domain_indicator,0,reduce=reduce[0])

    domain_maps = scatter(domain_reduced[data.domain_map_edge_index[0]],data.domain_map_edge_index[1],0,dim_size=data.num_targets,reduce=reduce[1])

    node_maps = scatter(x[data.node_map_edge_index[0]],data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[2])
    node_maps = scatter(node_maps,data.target.domain_indicator,0,dim_size=data.num_targets,reduce=reduce[3])
    return torch.cat([
        node_maps,
        domain_maps,
    ],-1)

def transfer1_1(x: Tensor, data: TransferData1, reduce: Union[list[str],str]='sum'):
    r"""for transfering from a ptensors1 to a ptensors1"""
    if isinstance(reduce,str):
        reduce = [reduce]*4

    x_intersect = x[data.node_map_edge_index[0]]
    x_intersect_reduced = scatter(x_intersect,data.intersect_indicator,0,reduce=reduce[0])
    x_domain = scatter(x,data.source.domain_indicator,0,reduce=reduce[1])
    x_invar = torch.cat([
        x_intersect_reduced, # local
        x_domain[data.domain_map_edge_index[0]] # global
    ],-1)
    y_local = scatter(
        torch.cat([
            x_intersect, # id
            x_invar[data.intersect_indicator] # (local,global)->local
            ],-1)
        ,data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[2])
    y_global_domain = scatter(
            x_invar,
            data.domain_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[3])
    return torch.cat([
        y_local,
        y_global_domain[data.target.domain_indicator]
    ],-1)