from __future__ import annotations
from typing import Any, NamedTuple, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data

class atomspack1(NamedTuple):
    atoms: Tensor
    r"""The atoms across all domains"""
    domain_indicator: Tensor
    num_domains: int
    
    def overlaps(self, other: atomspack1):

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
        return cls(atoms,domain_indicator,len(ls))
    
    def __str__(self) -> str:
        s = 'atoms:\n'
        for i in range(self.domain_indicator.max() + 1):
            s += f'\t{self.atoms[self.domain_indicator == i].tolist()}\n'
        return s

class atomspack2(atomspack1):
    atoms2: Tensor
    domain_indicator2: Tensor
    row_indicator: Tensor
    col_indicator: Tensor
    diag_idx: Tensor
    transpose_indicator: Tensor

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
    def from_atomspacks(cls, source: atomspack1, target: atomspack1):
        # computing nodes in the target domains that are intersected with.
        overlaps = source.overlaps(target)
        source_domains = source.domain_indicator[overlaps[0]]
        target_domains = target.domain_indicator[overlaps[1]]
        
        # 'intersect_indicator' represents the map from the source/target tensors to the intersections.
        domain_overlaps_edge_index, intersect_indicator = torch.stack([source_domains,target_domains]).unique(dim=1,return_inverse=True)
        num_nodes = max(source.atoms.max().item(),target.atoms.max().item()) + 1
        return cls(
            source,
            target,
            domain_overlaps_edge_index,
            overlaps,
            num_nodes,
            intersect_indicator)

class MultiTransferData1:
    r"""NOTE: we assume symmetry"""
    atomspacks: list[atomspack1]
    transfer_data: dict[tuple(int,int),TransferData1]
    num_nodes: int
    
    def __init__(self, atomspacks: list[atomspack1]) -> None:
        self.atomspacks = atomspacks
        self.transfer_data = dict[tuple(int,int),TransferData1]()
    
    def __getitem__(self, idx: tuple(int,int)):
        if idx in self.transfer_data:
            return self.transfer_data[idx]
        else:
            forward_map = TransferData1.from_atomspacks(self.atomspacks[idx[0]],self.atomspacks[idx[1]])
            backward_map = forward_map.reverse()
            self.transfer_data[idx] = forward_map
            self.transfer_data[[idx[1],idx[0]]] = backward_map
            return forward_map



class MultiScaleData(Data):
    edge_index_edge: Tensor

    edge_index_node_edge: Tensor
    edge_index_edge_cycle: Tensor
    
    edge_attr_cycle_edge: Tensor
    edge_attr_cycle: Tensor

    cycle_edge_cycle_indicator: Tensor
    r"""pass in a cycle/edge pair index and get back a cycle index."""

    edge_batch: Tensor
    cycle_batch: Tensor

    num_cycles: int

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index_edge':
            return torch.tensor(self.edge_index.size(1))
        elif key == 'edge_index_node_edge':
            return torch.tensor([[self.num_nodes],[self.edge_index.size(1)]])
        elif key == 'edge_index_edge_cycle':
            return torch.tensor([[self.edge_index.size(1)],[self.num_cycles]])
        elif key in ['edge_batch','cycle_batch']:
            if len(value) > 0:
                return value.max()
            else:
                return torch.tensor(1)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        return super().__cat_dim__(key, value, *args, **kwargs)
