from __future__ import annotations
from typing import Union
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Data
from objects import TransferData1, atomspack, atomspack2
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

# def transfer0_2(x: Tensor, domains: atomspack2, reduce: Union[list[str],str]='sum'):
#     # TODO: double check these linmaps.
#     if isinstance(reduce,str):
#         reduce = [reduce]*4
    
#     # 2 diagonal maps and 4 global maps.

#     y_sym = transfer0_1(x,domains,reduce[:3]) # used for maps where domain restrictions are the same for rows and columns.
#     global_sym = y_sym[domains.row_indicator]
    
#     diag = torch.zeros_like(global_sym)
#     diag[domains.diag_idx] = y_sym
    
    
#     return torch.cat([
#         y_sym[domains.diag_idx],
#     ],-1)