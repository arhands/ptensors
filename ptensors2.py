from __future__ import annotations
from typing import Union
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Data
from objects import TransferData1, TransferData2, atomspack1, atomspack2
from ptensors1 import linmaps1_0, linmaps0_1, transfer0_1

def linmaps0_2(x: Tensor, domains: atomspack2):
    full_broadcast = x[domains.atoms2]
    diag_broadcast = torch.zeros_like(full_broadcast)
    diag_broadcast[domains.diag_idx] = x[domains.atoms]
    return torch.cat([
        full_broadcast,
        diag_broadcast
    ],-1)


def linmaps2_0(x: Tensor, domains: atomspack2, reduce: Union[list[str],str]='sum'):
    if isinstance(reduce,str):
        reduce = [reduce]*2
    return torch.cat([
        scatter(x,domains.domain_indicator2,0,reduce=reduce[0]),
        scatter(x[domains.diag_idx],domains.domain_indicator,0,reduce=reduce[1]),
    ],-1)

def linmaps1_2_strict(x: Tensor, domains: atomspack2) -> list[Tensor]:
    r"""Excludes invariant maps (i.e., reductions that reduce to ptensors0 layers)."""
    row_broadcast = x[domains.row_indicator]
    col_broadcast = x[domains.col_indicator]
    diag_broadcast = torch.zeros_like(row_broadcast)
    diag_broadcast[domains.diag_idx] = x
    return [
        row_broadcast,
        col_broadcast,
        diag_broadcast,
    ]

def linmaps1_2(x: Tensor, domains: atomspack2, reduce: str='sum'):
    return torch.cat([
        *linmaps1_2_strict(x,domains),
        linmaps0_2(linmaps1_0(x,domains,reduce),domains)
    ],-1)

def linmaps2_1_strict(x: Tensor, domains: atomspack2, reduce: Union[list[str],str]='sum') -> list[Tensor]:
    r"""Excludes invariant maps (i.e., reductions that reduce to ptensors0 layers)."""
    if isinstance(reduce,str):
        reduce = [reduce]*2
    row_sum = scatter(x,domains.row_indicator,0,reduce=reduce[0])
    col_sum = scatter(x,domains.col_indicator,0,reduce=reduce[1])
    diag = x[domains.diag_idx]
    return [
        row_sum,
        col_sum,
        diag,
    ]

def linmaps2_1(x: Tensor, domains: atomspack2, reduce: Union[list[str],str]='sum'):
    if isinstance(reduce,str):
        reduce = [reduce]*4
    return torch.cat([
        linmaps2_1_strict(x,domains,reduce),
        linmaps0_1(linmaps2_0(x,domains,reduce[2:]),domains)
    ],-1)

def linmaps2_2(x: Tensor, domains: atomspack2, reduce: Union[list[str],str]='sum'):
    # TODO: double check these linmaps.
    if isinstance(reduce,str):
        reduce = [reduce]*4
    a = linmaps2_1_strict(x,domains,reduce)
    return torch.cat([
        *linmaps1_2_strict(a[0]),
        linmaps1_2(torch.cat(a[1:],-1)),# since the first two entries of 'a' reduce to the same value.
        x,
        x[domains.transpose_indicator],
    ],-1)

def transfer0_2_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the three linear maps that go to the intersecting region.
    """
    msgs = x[data.node_map_edge_index[0][data.intersect_indicator]]
    msgs_ij = msgs[data.ij_indicator]
    y_transpose = scatter(msgs_ij,data.node_pair_map_transpose[1],0,dim_size=len(data.target.atoms2),reduce=reduce[0])

    roots = x[data.ii_indicator]

    y = torch.cat([
        scatter(torch.cat([
            msgs_ij,
            roots,
        ],-1),data.node_pair_map_transpose[1],0,dim_size=len(data.target.atoms2),reduce=reduce[1]),
        y_transpose
        ],-1)
    return y

def transfer1_2_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the five linear maps in the intersecting region.
    """
    if isinstance(reduce,str):
        reduce = [reduce]*4
    msgs_i = x[data.node_map_edge_index[0]]
    msgs_inv = scatter(msgs_i,data.intersect_indicator,0,dim_size=len(data.target.atoms),reduce=reduce[0])
    msgs_i_inv = torch.cat([msgs_i,msgs_inv[data.intersect_indicator]],-1)
    msgs_ij_inv = msgs_i_inv[data.ij_indicator]


    # we exclude the invariant part to avoid duplication.
    y_ji = scatter(msgs_ij_inv[:,:x.size(1)],data.node_pair_map_transpose[1],0,dim_size=len(data.target.atoms2),reduce=reduce[1])
    # TODO: figure out if we actually save time by performing multiple smaller scatter operations instead of a few big ones.
    y_ij_inv = scatter(msgs_ij_inv,data.node_pair_map[1],0,dim_size=len(data.target.atoms2),reduce=reduce[2])

    y_i_inv = scatter(msgs_i_inv,data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[3])
    y_ii_inv = y_i_inv[data.target.diag_idx]

    y = torch.cat([
        y_ji,
        y_ii_inv,
        y_ij_inv,
        ],-1)
    return y

def transfer2_1_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the five linear maps in the intersecting region.
    """
    if isinstance(reduce,str):
        reduce = [reduce]*4
    msgs_ij = x[data.node_pair_map[0]]
    msgs_ji = x[data.node_pair_map_transpose[0]]
    msgs_ii = x[data.source.diag_idx[data.node_map_edge_index[0]]]
    msgs_ij_ji = torch.cat([msgs_ij,msgs_ji],-1)
    msgs_ij_ji_to_i = scatter(msgs_ij_ji,data.ij_indicator,0,reduce=reduce[0])
    msg_1 = torch.cat([msgs_ii,msgs_ij_ji_to_i],-1)
    msg_0 = scatter(msg_1[:,:-x.size(1)],
                       data.intersect_indicator,0,reduce=reduce[0])

    y = scatter(
        torch.cat([msg_1,msg_0[data.intersect_indicator]],-1)
        ,data.node_pair_map[1],0,dim_size=len(data.target.atoms2),reduce=reduce[1])
    
    return y

def transfer2_2_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the fifteen linear maps in the intersecting region.
    """
    if isinstance(reduce,str):
        reduce = [reduce]*5
    msg_ij = x[data.node_pair_map[0]]
    msg_ji = x[data.node_pair_map_transpose[0]]
    x_diag_i = x[data.source.diag_idx]
    
    # source diagonal
    msg_diag_i = x_diag_i[data.node_map_edge_index[0]]
    
    # reduced source diagonal
    msg_diag_inv = scatter(msg_diag_i,data.intersect_indicator,0,reduce=reduce[0])
    
    # reduction of rows and columns concatenated together
    msg_ij_ji = torch.cat([msg_ij,msg_ji],-1)
    msg_i_j = scatter(
        msg_ij_ji
        ,data.ij_indicator,0,reduce=reduce[1])
    
    # Reduction of full incoming intersecting ptensor region.
    msg_inv = scatter(msg_i_j[:,:x.size(1)],data.intersect_indicator,0,reduce=reduce[2])
    
    # all zeroth order incoming messages (2 total)
    msg_0 = torch.cat([msg_diag_inv,msg_inv],-1)

    # all zeroth and first order incoming messages (5 total)
    msg_01 = torch.cat([
        msg_i_j, # 2 maps
        msg_diag_i, # 1 map
        msg_0[data.intersect_indicator] # 2 maps
    ])

    y_01 = scatter(msg_01,data.domain_map_edge_index,0,dim_size=len(data.target.atoms),reduce=reduce[3])
    y_1 = y_01[:,:3*x.size(1)] # 3 maps
    
    # both strictly second order maps
    y_2 = scatter(msg_ij_ji,data.node_pair_map[1],0,dim_size=len(data.target.atoms2),reduce=reduce[4])

    y_012 = torch.cat([
        y_01[data.target.diag_idx], # 5 maps
        y_01[data.target.row_indicator], # 5 maps
        y_1[data.target.col_indicator], # 3 maps
        y_2 # 2 maps
    ],-1)
    return y_012

