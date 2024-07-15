"""
objects used for second order interactions.
"""
from __future__ import annotations
from typing import Optional, Union
import torch
from torch import Tensor
from objects1 import atomspack1, TransferData1
def _get_row_col_indicator(subgraphs: list[Tensor]|list[list]) -> tuple[Tensor,Tensor]:
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
def _get_transpose_indicator(subgraphs: list[Tensor]|list[list]) -> Tensor:
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
    row_indicator: Tensor
    """An indicator for broadcasting a 1st order layer into the rows of a second order layer."""
    col_indicator: Tensor
    """An indicator for broadcasting a 1st order layer into the columns of a second order layer."""
    def __init__(self, atoms, domain_indicator, num_domains, row_indicator, col_indicator) -> None:
        super().__init__(atoms, domain_indicator, num_domains)
        self.row_indicator = row_indicator
        self.col_indicator = col_indicator
    def get_num_atoms2(self):
        return len(self.row_indicator)
    def to(self,device):
        super().to(device)
        self.row_indicator = self.row_indicator.to(device)
        self.col_indicator = self.col_indicator.to(device)
    @classmethod
    def from_tensor_list(cls, ls: list[Tensor]):
        inst = atomspack1.from_tensor_list(ls)
        if len(ls) > 0:
            # constructs the elements in the atomspack1
            rows, cols = _get_row_col_indicator(ls)
        else:
            rows = torch.empty(0,dtype=torch.int64)
            cols = torch.empty(0,dtype=torch.int64)

        return cls(
            domain_indicator = inst.domain_indicator,
            atoms = inst.atoms,
            num_domains = inst.num_domains,
            row_indicator = rows,
            col_indicator = cols,
        )
    



class atomspack2(atomspack2_minimal):
    """Stores information for linmaps up to order 2."""
    diag_idx: Tensor
    """Gives the indices of the diagonal entries along the flattened 2nd order P-Tensor layer."""
    transpose_indicator: Tensor
    """For each P-tensor i and entry (j,k), gives the map (i,j,k) -> (i,k,j) flattened."""
    def __init__(self, atoms, domain_indicator, num_domains, row_indicator, col_indicator, diag_idx, transpose_indicator) -> None:
        super().__init__(atoms, domain_indicator, num_domains, row_indicator, col_indicator)
        self.diag_idx = diag_idx
        self.transpose_indicator = transpose_indicator
    def to(self,device):
        super().to(device)
        self.diag_idx = self.diag_idx.to(device)
        self.transpose_indicator = self.transpose_indicator.to(device)
    @classmethod
    def from_tensor_list(cls, ls: list[Tensor]):
        inst = atomspack2_minimal.from_tensor_list(ls)
        rows, cols = inst.row_indicator, inst.col_indicator
        if len(ls) > 0:
            # constructs the elements in the atomspack1
            diag_idx = (rows == cols).argwhere()[:,0]
            transpose = _get_transpose_indicator(ls)

        else:
            diag_idx = torch.empty(0,dtype=torch.int64)
            transpose = torch.empty(0,dtype=torch.int64)
        

        return cls(
            domain_indicator = inst.domain_indicator,
            atoms = inst.atoms,
            num_domains = inst.num_domains,
            row_indicator = rows,
            col_indicator = cols,
            diag_idx = diag_idx,
            transpose_indicator = transpose,
        )
    
    # the following are things that we only need for preprocessing.
    _atoms2: Optional[Tensor] = None
    r"""An indexing for the tuples of atoms. 
        Meaning, for every pair of atom indices (i,j),
        is the value in _atoms2[k] = i*num_nodes + j
        iff k in the 2nd order rep corrosponds to an
        occurence of (i,j).
    """
    _domains_indicator2: Optional[Tensor] = None
    """
    For each entry i in the flatted 2nd order P-Tensor layer, gives the corresponding
    P-Tensor ID. Meaning, entries i and j come from the same P-Tensor iff
        _domains_indicator2[i] == _domains_indicator2[j].
    """
    
    def get_atoms2(self, num_nodes: Optional[int] = None) -> Tensor:
        r"""An indexing for the tuples of atoms. 
            Meaning, for every pair of atom indices (i,j),
            is the value in _atoms2[k] = i*num_nodes + j
            iff k in the 2nd order rep corrosponds to an
            occurence of (i,j).
        """
        if self._atoms2 is None:
            assert num_nodes is not None
            self._atoms2 = num_nodes*self.atoms[self.row_indicator] + self.atoms[self.col_indicator]
        return self._atoms2
            

    def get_domains_indicator2(self) -> Tensor:
        """
        For each entry i in the flatted 2nd order P-Tensor layer, gives the corresponding
        P-Tensor ID. Meaning, entries i and j come from the same P-Tensor iff
            _domains_indicator2[i] == _domains_indicator2[j].
        """
        # TODO: Is it actually worth caching this??
        if self._domains_indicator2 is None:
            self._domains_indicator2 = self.domain_indicator[self.col_indicator]
        return self._domains_indicator2

class TransferData2(TransferData1):
    """
    Stores information needed to perform transfer maps up to order 2.
    """
    node_pair_map : Tensor
    """2nd order version of 'node_map_edge_index' for mapping nodes between second order representations."""

    ij_indicator: Tensor
    """indicator for mapping from a first order message rep to standard nodes in 2nd order messages."""
    source : atomspack2#type: ignore
    target : atomspack2#type: ignore
    def to(self,device):
        super().to(device)
        self.node_pair_map = self.node_pair_map.to(device)
        self.ij_indicator = self.ij_indicator.to(device)
    def __init__(self, source, target, domain_map_edge_index, node_map_edge_index, num_nodes, intersect_indicator, node_pair_map, ij_indicator):
        super().__init__(source, target, domain_map_edge_index, node_map_edge_index, num_nodes, intersect_indicator)
        self.node_pair_map = node_pair_map
        self.ij_indicator = ij_indicator

    @classmethod
    def from_atomspacks(cls, source: atomspack2, target: atomspack2, ensure_sources_subgraphs: bool):#type: ignore
        r"""
            On input of source and target atomspacks, generates a second order transfer map.
            'ensure_sources_subgraphs' makes it so that if we map from domain A\in source to domain B\in target,
            then A must be a subset of B.
        """
        sub = TransferData1.from_atomspacks(source,target,ensure_sources_subgraphs)
        sub = TransferData1.from_atomspacks(source,target,ensure_sources_subgraphs)
        
        source_reduction = atomspack1(source.get_atoms2(sub.num_nodes),source.get_domains_indicator2(),source.num_domains)
        target_reduction = atomspack1(target.get_atoms2(sub.num_nodes),target.get_domains_indicator2(),target.num_domains)
        transfer_data1_reduction = TransferData1.from_atomspacks(source_reduction,target_reduction,ensure_sources_subgraphs)
        
        return cls(
            source, 
            target, 
            sub.domain_map_edge_index, 
            sub.node_map_edge_index, 
            sub.num_nodes, 
            sub.intersect_indicator, 
            transfer_data1_reduction.node_map_edge_index, 
            transfer_data1_reduction.intersect_indicator
        )