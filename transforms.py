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
        edge_mask = edge_index[0] < edge_index[1]
        inc_edge_index = edge_index[:,edge_mask]
        data.edge_attr = data.edge_attr
        del data.edge_index
        data.edge_attr = data.edge_attr[edge_mask]
        # an indicator for mapping features to node->node messages.
        num_edges = len(data.edge_attr)
        edges = atomspack1(inc_edge_index.transpose(1,0).flatten(),torch.arange(inc_edge_index.size(1)).repeat_interleave(2),inc_edge_index.size(1))
        data.node2edge_index = torch.stack([
            edges.atoms,
            edges.domain_indicator
        ],-1)
        data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)

        # getting cycle related maps
        cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes))
        data.num_cycles = len(cycles)

        data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)
        if len(cycles) > 0:

            cycles = [c.to_list() for c in cycles]
            cycles_ap = atomspack1.from_list(cycles)

            
            edges_to_cycles : TransferData1 = TransferData1.from_atomspacks(edges,cycles_ap)
            data.edge2cycle_index = edges_to_cycles.domain_map_edge_index

            edge_attr_cycle = _get_cycle_attr(data,cycles_ap)

            data.cycle_attr = edge_attr_cycle
            
            data.cycle_batch = torch.zeros(len(cycles),dtype=torch.int64)
            
        else:
            
            data.edge2cycle_index = torch.empty(2,0,dtype=torch.int64)

            data.cycle_attr = torch.empty(0,dtype=torch.int64)
            
            data.cycle_batch = torch.zeros(0,dtype=torch.int64)

        return data