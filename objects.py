from __future__ import annotations
from typing import NamedTuple, Union
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Data

class atomspack(NamedTuple):
    atoms: Tensor
    r"""The atoms across all domains"""
    domain_indicator: Tensor
    
    def overlaps(self, other: atomspack):

        # NOTE: This is not super efficient, but python overhead is O(1)!
        incidence = self.atoms.unsqueeze(1) == other.atoms.unsqueeze(0)
        incidence = incidence.to_sparse_coo().coalesce()
        return incidence.indices()
        # return torch.stack([
        #     torch.where(incidence.any(1))[0],
        #     torch.where(incidence.any(0))[0],
        # ])
    
    @classmethod
    def from_list(cls, ls: list[list[int]]):
        atoms = torch.cat([torch.tensor(v) for v in ls])
        domain_indicator = torch.cat([torch.tensor([idx]).broadcast_to(len(v)) for idx, v in enumerate(ls)])
        return cls(atoms,domain_indicator)
    
    def __str__(self) -> str:
        s = 'atoms:\n'
        for i in range(self.domain_indicator.max() + 1):
            s += f'\t{self.atoms[self.domain_indicator == i].tolist()}\n'
        return s

class atomspack2(atomspack):
    atoms2: Tensor
    domain_indicator2: Tensor
    row_indicator: Tensor
    col_indicator: Tensor
    diag_idx: Tensor
    transpose_indicator: Tensor

class TransferData0:
    source: atomspack
    target: atomspack

    domain_map_edge_index: Tensor
    num_sources: int
    num_targets: int

    def __init__(self,source,target,domain_map_edge_index,num_sources,num_targets):
        self.source = source
        self.target = target
        self.domain_map_edge_index = domain_map_edge_index
        self.num_sources = num_sources
        self.num_targets = num_targets

class TransferData1(TransferData0):
    node_map_edge_index: Tensor
    num_nodes: int
    intersect_indicator: Tensor
    def __init__(self,source,target,domain_map_edge_index,num_sources,num_targets,node_map_edge_index,num_nodes,intersect_indicator):
        super().__init__(source,target,domain_map_edge_index,num_sources,num_targets)
        self.node_map_edge_index = node_map_edge_index
        self.num_nodes = num_nodes
        self.intersect_indicator = intersect_indicator
    @classmethod
    def from_atomspacks(cls, source: atomspack, target: atomspack):
        # computing nodes in the target domains that are intersected with.
        overlaps = source.overlaps(target)
        source_domains = source.domain_indicator[overlaps[0]]
        target_domains = target.domain_indicator[overlaps[1]]
        
        # 'intersect_indicator' represents the map from the source/target tensors to the intersections.
        domain_overlaps_edge_index, intersect_indicator = torch.stack([source_domains,target_domains]).unique(dim=1,return_inverse=True)
        num_sources = source.domain_indicator[-1].item() + 1
        num_targets = target.domain_indicator[-1].item() + 1
        num_nodes = max(source.atoms.max().item(),target.atoms.max().item()) + 1
        return cls(
            source,
            target,
            domain_overlaps_edge_index,
            num_sources,
            num_targets,
            overlaps,
            num_nodes,
            intersect_indicator)

class TransferData2(NamedTuple):
    source: atomspack
    target: atomspack

    domain_map_edge_index: Tensor
    node_map_edge_index: Tensor
    num_sources: int
    num_targets: int
    num_nodes: int
    @classmethod
    def from_atomspacks(cls, source: atomspack, target: atomspack):
        # computing nodes in the target domains that are intersected with.
        overlaps = source.overlaps(target)
        domain_overlaps_edge_index = torch.stack([source.domain_indicator[overlaps[0]],target.domain_indicator[overlaps[1]]]).unique(dim=1)
        return cls(
            source,
            target,
            domain_overlaps_edge_index,
            overlaps,
            source.domain_indicator[-1].item() + 1,
            target.domain_indicator[-1].item() + 1,
            max(source.atoms.max().item(),target.atoms.max().item()) + 1)