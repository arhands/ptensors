"""
objects used for zeroth and first order interactions.
"""
from __future__ import annotations
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_add

class atomspack1:
    atoms: Tensor
    r"""The atoms across all domains"""
    domain_indicator: Tensor
    num_domains: int
    raw_num_domains: int|Tensor
    def to(self,device):
        self.atoms = self.atoms.to(device)
        self.domain_indicator = self.domain_indicator.to(device)
    def __init__(self,atoms,domain_indicator,num_domains:Tensor|int) -> None:
        self.atoms = atoms
        self.domain_indicator = domain_indicator
        self.raw_num_domains = num_domains
        if isinstance(num_domains,Tensor):
            num_domains = int(num_domains.sum().item())
        self.num_domains = num_domains
    def overlaps1(self, other: atomspack1, ensure_source_subgraphs: bool):
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
    # TODO: add utilization for the below implementation
    # def overlaps1(self, other: atomspack1):
    #     r"""ensure subgraphs: only include connects where the subgraphs in 'self' are subgraphs to those in 'other'."""
    #     if len(self.atoms) == 0 or len(other.atoms) == 0:
    #         return torch.empty(2,0,dtype=torch.int64)
    #     # self_unique_atoms, self_unique_inverse = self.atoms.unique(True,True)
    #     self_unique_atoms = self.atoms.unique()
    #     # other_unique_atoms, other_unique_inverse = other.atoms.unique(True,True)
    #     other_unique_atoms = other.atoms.unique()
    #     i = 0
    #     j = 0
    #     incidence_unique_indices = []
    #     while i < len(self_unique_atoms) and j < len(other_unique_atoms):
    #         if self_unique_atoms[i] > other_unique_atoms[j]:
    #             j += 1
    #         elif self_unique_atoms[i] < other_unique_atoms[j]:
    #             i += 1
    #         else:
    #             incidence_unique_indices.append(self_unique_atoms[i])
    #             i += 1
    #             j += 1
        
    #     # now, we iterate through the incidence indices:
    #     incidences = []
    #     for i in incidence_unique_indices:
    #         self_inc = torch.argwhere(self.atoms == i).flatten()
    #         other_inc = torch.argwhere(other.atoms == i).flatten()

    #         # getting every combination
    #         self_inc = self_inc.unsqueeze(-1).broadcast_to(self_inc.size(0),other_inc.size(0))
    #         other_inc = other_inc.unsqueeze(0).broadcast_to(self_inc.size())
    #         self_inc = self_inc.flatten()
    #         other_inc = other_inc.flatten()
    #         incidences.append(torch.stack([self_inc,other_inc]))
    #     incidences = torch.cat(incidences,-1)
    #     return incidences
    
    @classmethod
    def from_tensor_list(cls, ls: list[Tensor]):
        if len(ls) > 0:
            atoms = torch.cat(ls)
            domain_indicator = torch.cat([torch.tensor([idx]).broadcast_to(len(v)) for idx, v in enumerate(ls)])
        else:
            atoms = torch.empty(0,dtype=torch.int64)
            domain_indicator = torch.empty(0,dtype=torch.int64)
        return cls(atoms,domain_indicator,len(ls))

    @classmethod
    def from_list(cls, ls: list[list[int]]):
        return cls.from_tensor_list([torch.tensor(v) for v in ls])
    
    def __str__(self) -> str:
        s = 'atoms:\n'
        for i in range(self.domain_indicator.max() + 1):
            s += f'\t{self.atoms[self.domain_indicator == i].tolist()}\n'
        return s

class TransferData0:
    source: atomspack1
    target: atomspack1

    domain_map_edge_index: Tensor
    r"""Represents an edge-index s.t. if P-Tensor i in the source layer maps to P-Tensor j in the target layer,
        then (and only then) [i,j]\in domain_map_edge_index.T
    """

    def __init__(self,source,target,domain_map_edge_index):
        self.source = source
        self.target = target
        self.domain_map_edge_index = domain_map_edge_index
    
    def to(self,device):
        self.source.to(device)
        self.target.to(device)
        self.domain_map_edge_index = self.domain_map_edge_index.to(device)

    def copy(self):
        r"""Creates a shallow copy of this object."""
        return TransferData0(self.source,self.target,self.domain_map_edge_index)

    def reverse(self, in_place: bool = False) -> TransferData0:
        if not in_place:
            return self.copy().reverse(True)
        
        self.source, self.target = self.target, self.source
        self.domain_map_edge_index = self.domain_map_edge_index.flip(0)
        
        return self
    @classmethod
    def from_atomspacks(cls, source: atomspack1, target: atomspack1, ensure_sources_subgraphs: bool):
        # computing nodes in the target domains that are intersected with.
        overlaps: Tensor = source.overlaps1(target,ensure_sources_subgraphs)
        source_domains: Tensor = source.domain_indicator[overlaps[0]]
        target_domains: Tensor = target.domain_indicator[overlaps[1]]
        
        domain_overlaps_edge_index: Tensor = torch.stack([source_domains,target_domains]).unique(dim=1)
        
        return cls(
            source,
            target,
            domain_overlaps_edge_index)

class TransferData1(TransferData0):
    node_map_edge_index: Tensor
    r"""
    If some node i appears in P-Tensors j in the source layer and k in the target layer and we map
    from j to k, then node_map_edge_index will have an edge incidendent from where i appears in the 
    j in the source 1st order P-tensor layer to where it appears in the k in the 1st order target P-Tensor layer.
    """
    intersect_indicator: Tensor
    r"""
    When mapping a pair of P-Tensors i and j in some P-Tensor layer, one can consider
    the node-wise maps given by 'node_map_edge_index' and the P-Tensor wise maps given
    by 'domain_map_edge_index'. 'intersect_indicator' broadcasts the intermediate messages from
    the domain-wise messages to the node-wise messages.
    """
    num_nodes: int
    """The number of nodes in the original graph."""
    def __init__(self,source,target,domain_map_edge_index,node_map_edge_index,num_nodes,intersect_indicator):
        super().__init__(source,target,domain_map_edge_index)
        self.node_map_edge_index = node_map_edge_index
        self.num_nodes = num_nodes
        self.intersect_indicator = intersect_indicator
    
    def to(self,device):
        super().to(device)
        self.node_map_edge_index = self.node_map_edge_index.to(device)
        self.intersect_indicator = self.intersect_indicator.to(device)

    def copy(self) -> "TransferData1":
        return TransferData1(**self.__dict__)
    
    def reverse(self, in_place: bool = False) -> TransferData1:
        """Reverses the direction of the linear map."""
        if not in_place:
            return self.copy().reverse(True)
        
        super().reverse(True)
        self.node_map_edge_index = self.node_map_edge_index.flip(0)
        return self

    @classmethod
    def from_atomspacks(cls, source: atomspack1, target: atomspack1, ensure_sources_subgraphs: bool):
        r"""
            On input of source and target atomspacks, generates a second order transfer map.
            'ensure_sources_subgraphs' makes it so that if we map from domain A\in source to domain B\in target,
            then A must be a subset of B.
        """
        # computing nodes in the target domains that are intersected with.
        overlaps = source.overlaps1(target,ensure_sources_subgraphs)
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