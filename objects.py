from __future__ import annotations
from typing import Any, NamedTuple, Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_add

class atomspack1(NamedTuple):
    atoms: Tensor
    r"""The atoms across all domains"""
    domain_indicator: Tensor
    num_domains: int
    
    def overlaps(self, other: atomspack1, ensure_source_subgraphs: bool):
        r"""ensure subgraphs: only include connects where the subgraphs in 'self' are subgraphs to those in 'other'."""
        if len(self.atoms) == 0 or len(other.atoms) == 0:
            return torch.empty(2,0,dtype=torch.int64)
        # NOTE: This is not super efficient, but python overhead is O(1)!
        incidence = self.atoms.unsqueeze(1) == other.atoms.unsqueeze(0)
        if ensure_source_subgraphs:
            incidence_domains = scatter_add(incidence.int(),self.domain_indicator,0)
            incidence_domains = scatter_add(incidence_domains,other.domain_indicator,1)
            source_domain_size = scatter_add(torch.ones_like(self.domain_indicator),self.domain_indicator,0)
            incidence_mask_domain = incidence_domains == source_domain_size.unsqueeze(1)
            incidence_mask = incidence_mask_domain[self.domain_indicator][:,other.domain_indicator]

            incidence[~incidence_mask] = False
        incidence = incidence.to_sparse_coo().coalesce()
        return incidence.indices()
        # return torch.stack([
        #     torch.where(incidence.any(1))[0],
        #     torch.where(incidence.any(0))[0],
        # ])
    
    @classmethod
    def from_list(cls, ls: list[list[int]]):
        if len(ls) > 0:
            atoms = torch.cat([torch.tensor(v) for v in ls])
            domain_indicator = torch.cat([torch.tensor([idx]).broadcast_to(len(v)) for idx, v in enumerate(ls)])
        else:
            atoms = torch.empty(0,dtype=torch.int64)
            domain_indicator = torch.empty(0,dtype=torch.int64)
        return cls(atoms,domain_indicator,len(ls))
    
    def __str__(self) -> str:
        s = 'atoms:\n'
        for i in range(self.domain_indicator.max() + 1):
            s += f'\t{self.atoms[self.domain_indicator == i].tolist()}\n'
        return s

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

class atomspack2(atomspack2_minimal):
    atoms2: Tensor
    domain_indicator2: Tensor
    # row_indicator: Tensor
    # col_indicator: Tensor
    diag_idx: Tensor
    transpose_indicator: Tensor

class atomspack3_minimal(atomspack2):
    r"""
    A minimal object for computing scaled dot product attention.
    """
    # atoms3: Tensor
    # # diagonals
    # ij_to_iij_indicator : Tensor
    # ij_to_iji_indicator : Tensor
    # ij_to_jii_indicator : Tensor
    
    ij_to_ijk_indicator : Tensor
    ij_to_ikj_indicator : Tensor
    # ij_to_kij_indicator : Tensor

class atomspack3_strick(atomspack3_minimal):
    r"""
    A minimal object for performing "strict" transforms between ptensor 2's and ptensor 3's.
    "strict" here means we only consider maps with 2nd order equivariance.
    """
    atoms3: Tensor
    # diagonals
    ij_to_iij_indicator : Tensor
    ij_to_iji_indicator : Tensor
    ij_to_jii_indicator : Tensor
    
    # ij_to_ijk_indicator : Tensor
    # ij_to_ikj_indicator : Tensor
    ij_to_kij_indicator : Tensor

class TransferData0:
    source: atomspack1
    target: atomspack1

    domain_map_edge_index: Tensor

    def __init__(self,source,target,domain_map_edge_index):
        self.source = source
        self.target = target
        self.domain_map_edge_index = domain_map_edge_index
    
    def copy(self):
        r"""Creates a shallow copy of this object."""
        return TransferData0(self.source,self.target,self.domain_map_edge_index)

    def reverse(self, in_place: bool = False) -> TransferData0:
        if not in_place:
            return self.copy().reverse(True)
        
        self.source, self.target = self.target, self.source
        self.domain_map_edge_index = self.domain_map_edge_index.flip(0)
        
        return self

class TransferData1(TransferData0):
    node_map_edge_index: Tensor
    intersect_indicator: Tensor
    num_nodes: int
    def __init__(self,source,target,domain_map_edge_index,node_map_edge_index,num_nodes,intersect_indicator):
        super().__init__(source,target,domain_map_edge_index)
        self.node_map_edge_index = node_map_edge_index
        self.num_nodes = num_nodes
        self.intersect_indicator = intersect_indicator
    
    def copy(self):
        return TransferData1(**self.__dict__)
    
    def reverse(self, in_place: bool = False) -> TransferData1:
        if not in_place:
            return self.copy().reverse(True)
        
        super().reverse(True)
        self.node_map_edge_index = self.node_map_edge_index.flip(0)
        return self

    @classmethod
    def from_atomspacks(cls, source: atomspack1, target: atomspack1, ensure_sources_subgraphs: bool):
        # computing nodes in the target domains that are intersected with.
        overlaps = source.overlaps(target,ensure_sources_subgraphs)
        source_domains = source.domain_indicator[overlaps[0]]
        target_domains = target.domain_indicator[overlaps[1]]
        
        # 'intersect_indicator' represents the map from the source/target tensors to the intersections.
        domain_overlaps_edge_index, intersect_indicator = torch.stack([source_domains,target_domains]).unique(dim=1,return_inverse=True)

        # if ensure_sources_subgraphs:
        #     overlap_size_source = scatter_add(torch.ones_like(intersect_indicator),intersect_indicator,0,dim_size=source.num_domains)
        #     source_size = scatter_add(torch.ones_like(source.domain_indicator),source.domain_indicator,0,dim_size=source.num_domains)
        #     mask_intersect_domains = overlap_size_source == source_size[domain_overlaps_edge_index[0]]
        #     mask_intersect_nodes = mask_intersect_domains[intersect_indicator]


        if source.num_domains > 0:
            num_nodes = source.atoms.max().item()
        else:
            num_nodes = 0

        if target.num_domains > 0:
            num_nodes = max(num_nodes,target.atoms.max().item())
        
        num_nodes += 1

        return cls(
            source,
            target,
            domain_overlaps_edge_index,
            overlaps,
            num_nodes,
            intersect_indicator)

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


class MultiScaleData(Data):
    node2edge_index: Tensor
    edge2cycle_index: Tensor
    node2cycle_index: Tensor

    edge_batch: Tensor
    cycle_batch: Tensor

    cycle_ind: Tensor

    cycle_attr: Tensor

    num_cycles: int

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'node2edge_index':
            return torch.tensor([[self.num_nodes],[len(self.edge_attr)]])
        elif key == 'edge2cycle_index':
            return torch.tensor([[len(self.edge_attr)],[self.num_cycles]])
        elif key == 'node2cycle_index':
            return torch.tensor([[self.num_nodes],[self.num_cycles]])
        elif key == 'cycle_ind':
            return self.num_nodes
        elif key == 'edge_batch':
            return value.max() + 1
        elif key == 'cycle_batch':
            if len(value) > 0:
                return value.max() + 1
            else:
                return torch.tensor(1)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class MultiScaleData_2(Data):
    node2edge_index: Tensor
    node2cycle_index: Tensor

    edge_batch: Tensor
    cycle_batch: Tensor
    
    # transfer data
    edge_atoms: Tensor
    edge_domain_indicator: Tensor
    num_edges: Union[int,Tensor]

    cycle_atoms: Tensor
    cycle_domain_indicator: Tensor
    num_cycles: Union[int,Tensor]
    
    edge2cycle_domain_map_edge_index: Tensor
    edge2cycle_node_map_edge_index: Tensor
    edge2cycle_intersect_indicator: Tensor
    edge2cycle_num_intersections: Union[int,Tensor]

    cycle2cycle_domain_map_edge_index: Tensor
    cycle2cycle_node_map_edge_index: Tensor
    cycle2cycle_intersect_indicator: Tensor
    cycle2cycle_num_intersections: Union[int,Tensor]
    
    def _get_num_cycles(self):
        if isinstance(self.num_cycles,int):
            return self.num_cycles
        return self.num_cycles.sum()
    def _get_num_edges(self):
        return len(self.edge_attr)
    def _get_edge2cycle_num_intersections(self):
        return self.edge2cycle_domain_map_edge_index.size(1)
    def _get_cycle2cycle_num_intersections(self):
        return self.cycle2cycle_domain_map_edge_index.size(1)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'node2edge_index':
            return torch.tensor([[self.num_nodes],[self._get_num_edges()]])
        elif 'atoms' in key:
            return self.num_nodes
        elif key == 'edge_domain_indicator':
            return self._get_num_edges()
        elif key == 'cycle_domain_indicator':
            return self._get_num_cycles()
        elif key == 'edge2cycle_node_map_edge_index':
            return torch.tensor([[len(self.edge_atoms)],[len(self.cycle_atoms)]],device=self.x.device)
        elif key == 'edge2cycle_domain_map_edge_index':
            return torch.tensor([[self._get_num_edges()],[self._get_num_cycles()]],device=self.x.device)
        elif key == 'edge2cycle_intersect_indicator':
            return self._get_edge2cycle_num_intersections()
        elif key == 'cycle2cycle_node_map_edge_index':
            return len(self.cycle_atoms)
        elif key == 'cycle2cycle_domain_map_edge_index':
            return self._get_num_cycles()
        elif key == 'cycle2cycle_intersect_indicator':
            return self._get_cycle2cycle_num_intersections()
        elif key == 'cycle_ind':
            return self.num_nodes
        elif key in ['cycle_batch','edge_batch']:
            if len(value) > 0:
                return value.max() + 1
            else:
                return torch.tensor(1,device=self.x.device)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def set_edge2cycle(self, edge2cycle: TransferData1):
        edges = edge2cycle.source
        cycles = edge2cycle.target
        self.num_cycles = cycles.num_domains
        self.cycle_atoms = cycles.atoms
        self.cycle_domain_indicator = cycles.domain_indicator
        
        self.num_edges = edges.num_domains
        self.edge_atoms = edges.atoms
        self.edge_domain_indicator = edges.domain_indicator
        
        self.edge2cycle_domain_map_edge_index = edge2cycle.domain_map_edge_index
        self.edge2cycle_node_map_edge_index = edge2cycle.node_map_edge_index
        self.edge2cycle_intersect_indicator = edge2cycle.intersect_indicator
        self.edge2cycle_num_intersections = edge2cycle.domain_map_edge_index.size(1)

        self.cycle_batch = torch.zeros(cycles.num_domains,dtype=torch.int64)
    
    def set_edge2cycle_4(self, edge2cycle: TransferData1):
        edges = edge2cycle.source
        cycles = edge2cycle.target
        self.num_cycles = cycles.num_domains
        self.cycle_atoms = cycles.atoms
        self.cycle_domain_indicator = cycles.domain_indicator
        
        self.num_edges = edges.num_domains
        self.edge_atoms = edges.atoms
        self.edge_domain_indicator = edges.domain_indicator
        
        self.edge2cycle_domain_map_edge_index = edge2cycle.domain_map_edge_index
        self.edge2cycle_node_map_edge_index = edge2cycle.node_map_edge_index
        self.edge2cycle_intersect_indicator = edge2cycle.intersect_indicator
        self.edge2cycle_num_intersections = edge2cycle.domain_map_edge_index.size(1)

        self.cycle_batch = torch.zeros(len(cycles.atoms),dtype=torch.int64)
    
    def get_edge2cycle(self):
        num_edges = self.num_edges if isinstance(self.num_edges,int) else torch.sum(self.num_edges).item()
        num_cycles = self.num_cycles if isinstance(self.num_cycles,int) else torch.sum(self.num_cycles).item()
        return TransferData1(
            source=atomspack1(
                atoms = self.edge_atoms,
                domain_indicator = self.edge_domain_indicator,
                num_domains = num_edges
            ),
            target=atomspack1(
                atoms = self.cycle_atoms,
                domain_indicator = self.cycle_domain_indicator,
                num_domains = num_cycles
            ),
            domain_map_edge_index=self.edge2cycle_domain_map_edge_index,
            node_map_edge_index=self.edge2cycle_node_map_edge_index,
            intersect_indicator=self.edge2cycle_intersect_indicator,
            num_nodes=self.num_nodes,
        )
    def set_cycle2cycle(self, cycle2cycle: TransferData1):
        r"""
        NOTE: this assumes you will use edge2cycle
        """
        
        self.cycle2cycle_domain_map_edge_index = cycle2cycle.domain_map_edge_index
        self.cycle2cycle_node_map_edge_index = cycle2cycle.node_map_edge_index
        self.cycle2cycle_intersect_indicator = cycle2cycle.intersect_indicator
        self.cycle2cycle_num_intersections = cycle2cycle.domain_map_edge_index.size(1)
    def get_cycle2cycle(self):
        num_cycles = self.num_cycles if isinstance(self.num_cycles,int) else torch.sum(self.num_cycles).item()
        cycles = atomspack1(
            atoms = self.cycle_atoms,
            domain_indicator = self.cycle_domain_indicator,
            num_domains = num_cycles
        )
        return TransferData1(
            source=cycles,
            target=cycles,
            domain_map_edge_index=self.cycle2cycle_domain_map_edge_index,
            node_map_edge_index=self.cycle2cycle_node_map_edge_index,
            intersect_indicator=self.cycle2cycle_intersect_indicator,
            num_nodes=self.num_nodes,
        )
