from __future__ import annotations
from typing import Any, NamedTuple, Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_add
from objects import TransferData1, atomspack1
# def _get_row_col_indicator(domain_indicator: Tensor) -> tuple[Tensor,Tensor]:
#     rows = []
#     cols = []
#     for i in domain_indicator.unique():
#         entries = (domain_indicator == i).argwhere()[:,0]
#         size = len(entries)
#         row = entries.unsqueeze(-1).broadcast_to(-1,size).flatten()
#         col = entries.unsqueeze(0).broadcast_to(size,-1).flatten()
#         rows.append(row)
#         cols.append(col)
#     return torch.cat(rows), torch.cat(cols)
def _get_row_col_indicator(subgraphs: list[Union[Tensor,list]]) -> tuple[Tensor,Tensor]:
    r"""
    NOTE: this assumes 'domain_indicator' is ordered the same as 'subgraphs'
        and increments domains by one starting at zero.
        This is how the code typically does it, but it's still good to 
        keep in mind. (the above commented out code is more general)
    """
    rows = []
    cols = []
    count = 0
    for subg in subgraphs:
        size = len(subg)
        ar = torch.arange(count,count + size)
        row = ar.unsqueeze(-1).broadcast_to(-1,size).flatten()
        col = ar.unsqueeze(0).broadcast_to(size,-1).flatten()
        rows.append(row)
        cols.append(col)
        count += size
    return torch.cat(rows), torch.cat(cols)
def _get_transpose_indicator(subgraphs: list[Union[Tensor,list]]) -> Tensor:
    r"""
    NOTE: this assumes 'domain_indicator' is ordered the same as 'subgraphs'
        and increments domains by one starting at zero.
        This is how the code typically does it, but it's still good to 
        keep in mind.
    """
    tensor_list = []
    count = 0
    for subg in subgraphs:
        size = len(subg)**2
        if size > 0:
            transpose = torch.arange(count,count + size).view(-1,len(subg)).transpose(1,0).flatten()
            count += size
            tensor_list.append(transpose)
    return torch.cat(tensor_list)
class atomspack2_minimal(atomspack1):
    r"""
    A minimal object for computing scaled dot product attention.
    """
    # atoms2: Tensor
    # domain_indicator2: Tensor
    row_indicator: Tensor
    col_indicator: Tensor
    # diag_idx: Tensor
    # transpose_indicator: Tensor
    def get_num_atoms2(self):
        return len(self.row_indicator)

    @classmethod
    def from_tensor_list(cls, ls: list[Tensor]):
        inst = atomspack1.from_tensor_list(ls)
        if len(ls) > 0:
            # constructs the elements in the atomspack1
            # rows, cols = _get_row_col_indicator(inst.domain_indicator)
            rows, cols = _get_row_col_indicator(ls)
        else:
            rows = torch.empty(0,dtype=torch.int64)
            cols = torch.empty(0,dtype=torch.int64)

        return cls(
            # domain_indicator = inst.domain_indicator,
            # atoms = inst.atoms,
            **inst,
            row_indicator = rows,
            col_indicator = cols,
        )
    


class atomspack2(atomspack2_minimal):
    atoms2: Tensor
    # domain_indicator2: Tensor
    # row_indicator: Tensor
    # col_indicator: Tensor
    diag_idx: Tensor
    transpose_indicator: Tensor
    
    @classmethod
    def from_tensor_list(cls, ls: list[Tensor]):
        inst = atomspack2_minimal.from_tensor_list(ls)
        if len(ls) > 0:
            # constructs the elements in the atomspack1
            rows, cols = inst.row_indicator, inst.col_indicator
            diag_idx = (rows == cols).argwhere()[:,0]
            transpose = _get_transpose_indicator(ls)

        else:
            diag_idx = torch.empty(0,dtype=torch.int64)
            transpose = torch.empty(0,dtype=torch.int64)

        return cls(
            **inst,
            diag_idx = diag_idx,
            transpose = transpose,
        )

class TransferData2(TransferData1):
    # NOTE: it is required that the messages in 'node_pair_map' and 'node_pair_map_transpose' are so that the messages and the transposed messages align.

    node_pair_map : Tensor
    """2nd order version of 'node_map_edge_index' for mapping nodes between second order representations."""

    # ii_indicator: Tensor
    # """indicator for mapping from a first order message rep to root nodes in 2nd order messages."""

    ij_indicator: Tensor
    """indicator for mapping from a first order message rep to standard nodes in 2nd order messages."""

    node_pair_map_transpose: Tensor
    """
        2nd order version of 'node_map_edge_index' for mapping nodes between second order representations.
        The idea is that this is the same as the 'node_pair_map' map, but it's transposed so that doing the first half of 'node_pair_map' and 
            following it up with this causes a transposition and visa-versa.
        This ensures reversability.
    """

    source : atomspack2
    target : atomspack2