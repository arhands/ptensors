from __future__ import annotations
from typing import Union
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Data
from objects import TransferData1, atomspack1
from ptensors1 import linmaps1_0, linmaps0_1, transfer0_1
from objects2 import TransferData2, atomspack2, atomspack2_minimal

def linmaps0_2(x: Tensor, domains: atomspack2):
    # full_broadcast = x[domains.atoms2]
    first_order_rep = x[domains.domain_indicator]
    diag_broadcast = torch.zeros(
        domains.get_num_atoms2(),x.size(1),dtype=x.dtype,device=x.device)
    diag_broadcast[domains.diag_idx] = first_order_rep
    return torch.cat([
        first_order_rep[domains.col_indicator],
        diag_broadcast
    ],-1)


def linmaps2_0(x: Tensor, domains: atomspack2, reduce: Union[list[str],str]='sum'):
    if isinstance(reduce,str):
        reduce = [reduce]*2
    return torch.cat([
        # scatter(x,domains.domain_indicator2,0,reduce=reduce[0]),
        scatter(x,domains.domain_indicator[domains.col_indicator],0,reduce=reduce[0]),
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

def transfer0_2_minimal(x: Tensor, data: TransferData2, reduce: str = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the two linear maps strictly in the intersecting region.
    """
    msgs = x[data.node_map_edge_index[0]]
    msgs_i = msgs[data.intersect_indicator]
    
    # first order rep of message
    y_i = scatter(msgs_i,data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce)
    return linmaps0_2(y_i,data.target)

def transfer1_2_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the five linear maps in the intersecting region.
    """
    if isinstance(reduce,str):
        reduce = [reduce]*2
    msgs_i = x[data.node_map_edge_index[0]]
    msgs_inv = scatter(msgs_i,data.intersect_indicator,0,reduce=reduce[0])
    msgs_i_inv = torch.cat([msgs_i,msgs_inv[data.intersect_indicator]],-1)
    y_i_inv = scatter(msgs_i_inv,data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[1])
    
    # Now we essentially do a strict_linmaps1_2, 
    #   but we have to avoid copying the "inv" part twice.
    domain = data.target
    cols = y_i_inv[domain.col_indicator,:x.size(1)] # one map
    diag = torch.zeros_like(rows)
    diag[domain.diag_idx] = y_i_inv # two maps
    rows = y_i_inv[domain.row_indicator] # two maps
    return torch.cat([
        cols,
        diag,
        rows,
    ],-1)

# def transfer2_1_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
#     r"""
#     Performs the minimum number of reductions such that the full space of linear maps 
#     is covered if a linmaps operation is performed before and after calling this.
#     NOTE: this only has the same representation power for commutative reductions.

#     We only consider the five linear maps in the intersecting region.
#     """
#     if isinstance(reduce,str):
#         reduce = [reduce]*4
#     msgs_ij = x[data.node_pair_map[0]]
#     msgs_ji = x[data.node_pair_map_transpose[0]]
#     msgs_ii = x[data.source.diag_idx[data.node_map_edge_index[0]]]
#     msgs_ij_ji = torch.cat([msgs_ij,msgs_ji],-1)
#     msgs_ij_ji_to_i = scatter(msgs_ij_ji,data.ij_indicator,0,reduce=reduce[0])
#     msg_1 = torch.cat([msgs_ii,msgs_ij_ji_to_i],-1)
#     msg_0 = scatter(msg_1[:,:-x.size(1)],
#                        data.intersect_indicator,0,reduce=reduce[0])

#     y = scatter(
#         torch.cat([msg_1,msg_0[data.intersect_indicator]],-1)
#         ,data.node_pair_map[1],0,dim_size=data.target.get_num_atoms2(),reduce=reduce[1])
    
#     return y

# def transfer2_2_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum', large_ptensors: bool = False):
def transfer2_2_minimal(x: Tensor, data: TransferData2, reduce: Union[str,list[str]] = 'sum'):
    r"""
    Performs the minimum number of reductions such that the full space of linear maps 
    is covered if a linmaps operation is performed before and after calling this.
    NOTE: this only has the same representation power for commutative reductions.

    We only consider the fifteen linear maps in the intersecting region.

    This is designed to be optimized for large ptensors.
    """
    if isinstance(reduce,str):
        reduce = [reduce]*5

    x_ji = x[data.source.transpose_indicator]
    x_ij_ji = torch.cat([x,x_ji],-1)
    x_ii = x[data.source.diag_idx]

    # both strictly second order messages
    msg_ij_ji = x_ij_ji[data.node_pair_map[0]]
    
    msg_ii = x_ii[data.node_map_edge_index[0]]
    
    # sum of rows and columns
    msg_i_j = scatter(msg_ij_ji,data.ij_indicator,0,dim_size=len(msg_ii),reduce=reduce[0])

    #all three first order messages
    msg_i_j_ii = torch.cat([msg_i_j,msg_ii],-1)

    #both zeroth order messages
    msg_inv = scatter(msg_i_j_ii[:,x.size(1):],data.intersect_indicator,0,reduce=reduce[1])
    
    msg_i_j_ii_inv = torch.cat([msg_i_j_ii,msg_inv[data.intersect_indicator]],-1)

    # if large_ptensors:
    # For the next part, we make the following assumption about computational efficiency:
    # We assume the individual ptensors are large and so we are better 
    # off performing the small scatter operations separately for each order.

    # TODO: figure out if alternate methods may actually give improvements for
    # reasonably small ptensors.
    # The only theoretical improvement is better parallelization
    #   for large ptensors and less ops for small ptensors.
    

    # Now, we combine all of the messages.
    y_2 = scatter(msg_ij_ji,data.node_pair_map[1],0,dim_size=data.target.get_num_atoms2(),reduce=reduce[1])
    y_10 = scatter(msg_i_j_ii_inv,data.node_map_edge_index[1],0,reduce=reduce[1])
    
    # We now raise y_10 to second order, avoiding duplication of the invariant parts.
    diag = torch.zeros(len(y_2),y_10.size(1),dtype=x.dtype,device=x.device)
    diag[data.target.diag_idx] = y_10
    return torch.cat([
        y_2,
        diag,
        y_10[data.target.row_indicator],
        y_10[data.target.col_indicator,:3*x.size(1)],
    ],-1)
    # else:
    #     # For the next part, we make the following assumption about computational efficiency:
    #     # the ptensors are small and so we want to minimize the number of scatter ops being performed.
    #     msg_i_j_ii_inv_ij_ji = torch.cat([
    #         msg_i_j_ii_inv[data.ij_indicator],
    #         msg_ij_ji
    #     ])
    #     y_i_j_ii_inv_ij_ji = scatter(msg_i_j_ii_inv_ij_ji,data.node_pair_map[1],0,dim_size=data.target.get_num_atoms2(),reduce=reduce[1])
        
    #     # the two remaining things to do are tranpsose i, j, & ii, and construct the diagonal.


