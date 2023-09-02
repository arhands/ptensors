from typing import Any, List, Union, Tuple, overload
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Batch, Data
from induced_cycle_finder import from_edge_index, get_induced_cycles
import torch
from torch import Tensor
import ptens as p
from utils import GraphMapCache

class ProcessedData(Data):
    
class OnMoveToDeviceTransform(BaseTransform):
    def __init__(self, patterns: List[str], target_device: str, force_clone: bool = False) -> None:
        super().__init__()
        self.force_clone = force_clone
        self.target_device = target_device

        
    @overload
    def __call__(self, data: Batch) -> Tuple[Batch,GraphMapCache]:...
    @overload
    def __call__(self, data: Data) -> Tuple[Data,GraphMapCache]:...
    
    def __call__(self, data: Union[Batch,Data]) -> Tuple[Union[Batch,Data],GraphMapCache]:
        if self.force_clone:
            data = data.clone()

        num_nodes = data.num_nodes
        edge_index = data.edge_index
        induced_cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes))
        induced_cycles = [v.to_list() for v in induced_cycles]

        G = p.graph.from_edge_index(data.edge_index.float(),data.num_nodes)
        map_cache = GraphMapCache(G,{'E' : G.edges(), 'cycles' : induced_cycles})
        return data.cuda(), map_cache


# class AddThreeSkeleton(BaseTransform):
#     def __call__(self, data: Union[Batch,Data]) -> Union[Batch,Data]:
#         num_nodes = data.num_nodes
#         edge_index = data.edge_index
#         induced_cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes))
#         induced_cycles = [torch.tensor(v.to_list()) for v in induced_cycles]
        
#         vertex_indicator = torch.cat([torch.tensor(v) for v in induced_cycles])
#         cycle_indicator = torch.cat([torch.tensor(idx).broadcast_to(len(v)) for idx, v in enumerate(induced_cycles)])
#         to_cycle_edge_index = torch.stack([
#             torch.cat([torch.tensor(idx).broadcast_to(len(v)) for idx, v in enumerate(induced_cycles)]),
#             torch.cat([torch.tensor(v) for v in induced_cycles])
#         ])
#         return super().__call__(data)