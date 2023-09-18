from __future__ import annotations
from typing import Any, NamedTuple, Optional, Union
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
    
    def _get_num_cycles(self):
        if isinstance(self.num_cycles,int):
            return self.num_cycles
        return self.num_cycles.sum()
    def _get_num_edges(self):
        return len(self.edge_attr)
    def _get_edge2cycle_num_intersections(self):
        return self.edge2cycle_domain_map_edge_index.size(1)

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
            return torch.tensor([[len(self.edge_atoms)],[len(self.cycle_atoms)]])
        elif key == 'edge2cycle_domain_map_edge_index':
            return torch.tensor([[self._get_num_edges()],[self._get_num_cycles()]])
        elif key == 'edge2cycle_intersect_indicator':
            return self._get_edge2cycle_num_intersections()
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
