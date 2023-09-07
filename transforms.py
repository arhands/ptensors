from typing import Any, Iterable, List, Union, Tuple, overload
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Batch, Data
from induced_cycle_finder import from_edge_index, get_induced_cycles
import torch
from torch import Tensor
from objects import TransferData1, MultiScaleData, atomspack1
from induced_cycle_finder import get_induced_cycles, from_edge_index
from torch_scatter import scatter_sum

def _get_cycle_attr(data: Data, cycles: atomspack1):
    return scatter_sum(torch.nn.functional.one_hot(data.x,22)[cycles.atoms],cycles.domain_indicator,0,dim_size=cycles.num_domains)

class PreprocessTransform(BaseTransform):
    def __call__(self, data_: Data) -> MultiScaleData:
        data = MultiScaleData()
        data.__dict__.update(data_.__dict__)
        
        data.x = data.x.flatten()
        data.y = data.y.flatten().view(-1,1)
        data.edge_attr = data.edge_attr.flatten()
        edge_index : Tensor = data.edge_index
        num_nodes : int = data.num_nodes

        # first, we compute maps between edges and nodes.
        # NOTE: we assume graph is simple and undirected.
        data.edge_attr = data.edge_attr

        # an indicator for mapping features to node->node messages.
        num_edges = len(data.edge_attr)
        data.edge2node_msg_ind = torch.arange(num_edges).tile(2)

        data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)

        # getting cycle related maps
        cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes))
        data.num_cycles = len(cycles)

        if len(cycles) > 0:

            edges = atomspack1(edge_index.transpose(1,0).flatten(),torch.arange(edge_index.size(1)),edge_index.size(1))
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
            
            # node_counts = torch.tensor([len(c) for c in cycles])
            edge_attr_cycle = _get_cycle_attr(data,cycles_ap)
            edge_counts_squared = edge_counts**2
            
            edge_pair_cycle_indicator = torch.arange(len(cycles)).repeat_interleave(edge_counts_squared) # needed for mapping cycles to cycle-edge pairs.

            data.edge2edge_edge_index = edge_index_edge
            data.edge2cycle_edge_index = edge_index_edge_cycle

            data.cycle_attr = edge_attr_cycle
            
            data.cycle2edge_msg_ind = edge_pair_cycle_indicator

            data.cycle_batch = torch.zeros(len(cycles),dtype=torch.int64)
            
        else:
            
            data.edge2edge_edge_index = torch.empty(2,0,dtype=torch.int64)
            data.edge2cycle_edge_index = torch.empty(2,0,dtype=torch.int64)

            data.cycle_attr = torch.empty(0,dtype=torch.int64)
            
            data.cycle2edge_msg_ind = torch.empty(0,dtype=torch.int64)

            data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)
            data.cycle_batch = torch.zeros(0,dtype=torch.int64)
            data.num_cycles = 0

        return data