from typing import Any, Iterable, List, Union, Tuple, overload
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Batch, Data
from induced_cycle_finder import from_edge_index, get_induced_cycles
import torch
from torch import Tensor
from objects import TransferData1, MultiScaleData, atomspack1
from induced_cycle_finder import get_induced_cycles, from_edge_index

class PreprocessTransform(BaseTransform):
    def __call__(self, data_: Data) -> MultiScaleData:
        data = MultiScaleData()
        data.__dict__.update(data_.__dict__)
        
        data.x = data.x.flatten()
        data.edge_attr = data.edge_attr.flatten()

        edge_index : Tensor = data.edge_index
        num_nodes : int = data.num_nodes
        cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes))
        # nodes = atomspack1(torch.arange(num_nodes),torch.arange(num_nodes),num_nodes)
        edges = atomspack1(edge_index.transpose(1,0).flatten(),torch.arange(edge_index.size(1)).repeat_interleave(2),edge_index.size(1))
        
        edge_index_node_edge = torch.stack([edge_index[0],torch.arange(edges.num_domains)],0)
        data.edge_index_node_edge = edge_index_node_edge
        
        if len(cycles) > 0:

            
            cycles = [c.to_list() for c in cycles]
            cycles_ap = atomspack1.from_list(cycles)

            
            edges_to_cycles : TransferData1 = TransferData1.from_atomspacks(edges,cycles_ap)
            edge_index_edge_cycle = edges_to_cycles.domain_map_edge_index
            cycle_cliques = []
            # lens = torch.tensor([len(c) for c in cycles])
            edge_counts = []
            # cycles = [torch.tensor(c) for c in cycles]
            for i in range(len(cycles)):
                incoming_edges = edges_to_cycles.domain_map_edge_index[0][edges_to_cycles.domain_map_edge_index[1] == i]
                length = len(incoming_edges)
                cycle_cliques.append(torch.stack([
                    incoming_edges.repeat(length),
                    torch.repeat_interleave(incoming_edges,length)
                ]))
                edge_counts.append(length)
            edge_index_edge = torch.cat(cycle_cliques,-1)
            edge_counts = torch.tensor(edge_counts)
            # lens2 = lens**2


            # computing indicator for cycle size
            # TODO: figure out better features for cycles.
            
            node_counts = torch.tensor([len(c) for c in cycles])
            edge_attr_cycle = node_counts
            edge_counts_squared = edge_counts**2
            edge_attr_cycle_edge = node_counts.repeat_interleave(edge_counts_squared)
            
            edge_pair_cycle_indicator = torch.arange(len(cycles)).repeat_interleave(edge_counts_squared) # needed for mapping cycles to cycle-edge pairs.

            data.edge_index_edge = edge_index_edge
            data.edge_index_edge_cycle = edge_index_edge_cycle

            data.edge_attr_cycle_edge = edge_attr_cycle_edge
            data.edge_attr_cycle = edge_attr_cycle[edge_pair_cycle_indicator]
            
            data.cycle_edge_cycle_indicator = edge_pair_cycle_indicator

            data.edge_batch = torch.zeros(edges.num_domains)
            data.cycle_batch = torch.zeros(len(cycles))
            data.num_cycles = len(cycles)
        else:
            
            data.edge_index_edge = torch.empty(2,0,dtype=torch.int64)
            data.edge_index_edge_cycle = torch.empty(2,0,dtype=torch.int64)

            data.edge_attr_cycle_edge = torch.empty(0,dtype=torch.int64)
            data.edge_attr_cycle = torch.empty(0,dtype=torch.int64)
            
            data.cycle_edge_cycle_indicator = torch.empty(0,dtype=torch.int64)

            data.edge_batch = torch.zeros(edges.num_domains)
            data.cycle_batch = torch.zeros(0)
            data.num_cycles = 0

        return data