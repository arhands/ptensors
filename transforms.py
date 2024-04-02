import math
from typing import Any, Iterable, List, Literal, Union, Tuple, overload
from torch_geometric.transforms import BaseTransform, Compose, OneHotDegree, Constant
from torch_geometric.utils import to_undirected, degree
from torch_geometric.data import Batch, Data
from torch_geometric.typing import OptTensor
from induced_cycle_finder import from_edge_index, get_induced_cycles
import torch
from torch.nn import Identity
from torch import Tensor
from objects import TransferData1, MultiScaleData, atomspack1, MultiScaleData_2
from induced_cycle_finder import get_induced_cycles, from_edge_index
from data_handler import dataset_type, _tu_datasets, tu_dataset_type

class PreprocessTransform_old(BaseTransform):
    def __init__(self, max_cycle_size: Union[int,float] = math.inf) -> None:
        super().__init__()
        self.max_cycle_size = max_cycle_size

    def __call__(self, data_: Data) -> MultiScaleData:
        data = MultiScaleData()
        data.__dict__.update(data_.__dict__)
        edge_index : Tensor = data.edge_index
        num_nodes : int = data.num_nodes

        # first, we compute maps between edges and nodes.
        # NOTE: we assume graph is simple and undirected.
        edge_mask = edge_index[0] < edge_index[1]
        inc_edge_index = edge_index[:,edge_mask]
        data.edge_attr = data.edge_attr
        del data.edge_index
        data.edge_attr = data.edge_attr[edge_mask]

        num_edges = len(data.edge_attr)
        assert num_edges > 0, data.edge_attr.size()
        edges = atomspack1(inc_edge_index.transpose(1,0).flatten(),torch.arange(inc_edge_index.size(1)).repeat_interleave(2),inc_edge_index.size(1))
        data.node2edge_index = torch.stack([
            edges.atoms,
            edges.domain_indicator
        ])
        data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)
        # getting cycle related maps
        
        cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes),self.max_cycle_size)
        data.num_cycles = len(cycles)

        if len(cycles) > 0:

            cycles = [c.to_list() for c in cycles]
            cycles_ap = atomspack1.from_list(cycles)

            
            edges_to_cycles : TransferData1 = TransferData1.from_atomspacks(edges,cycles_ap,True)
            data.edge2cycle_index = edges_to_cycles.domain_map_edge_index
            
            data.cycle_batch = torch.zeros(len(cycles),dtype=torch.int64)
            
            data.node2cycle_index = torch.stack([cycles_ap.atoms,cycles_ap.domain_indicator])
        else:
            
            data.edge2cycle_index = torch.empty(2,0,dtype=torch.int64)
            data.node2cycle_index = torch.empty(2,0,dtype=torch.int64)

            data.cycle_attr = torch.empty(0,dtype=torch.int64)
            
            data.cycle_batch = torch.zeros(0,dtype=torch.int64)

        data.num_nodes = len(data.x)
        return data

class GraphPropertyPreprocessor(BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.edge_attr = torch.empty(data.edge_index.size(1),dtype=torch.int8) # since we only care about the size.
        return data

class GraphPropertyProcessor(BaseTransform):
    def __init__(self,target: Literal[0,1,2]) -> None:
        super().__init__()
        self.target = target
    def __call__(self, data: Any) -> Any:
        data.y = data.y[:,self.target]
        return data

class HIVPreprocessor(BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.y = data.y.float()
        return data

class ZINCPreprocessor(BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.x = data.x.flatten()
        data.y = data.y.unsqueeze(-1)
        data.edge_attr = data.edge_attr.flatten()
        return data 
# idx=0
class TUDatasetPreprocessor(BaseTransform):
    def __init__(self, ds: str) -> None:
        super().__init__()
        self.is_multilabel = ds in ['COLLAB','IMDB-MULTI','ENZYMES']
        self.ignore_degree = ds == 'REDDIT_BINARY'
    def __call__(self, data):
        # global idx
        # idx += 1
        # print(idx,flush=True)
        if data.x is not None:
            data.x = data.x.argmax(1)
        else:
            # data.x = torch.empty(data.num_nodes,dtype=torch.int8) # since we only care about the size.
            if self.ignore_degree:
                data.x = torch.zeros(data.num_nodes,dtype=torch.int8)
            else:
                data.x = degree(data.edge_index[0],data.num_nodes,dtype=torch.int32)
        if data.edge_attr is not None:
            if data.edge_attr.ndim > 1:
                data.edge_attr = data.edge_attr.argmax(1)
        else:
            if self.ignore_degree:
                data.edge_attr = torch.zeros(data.edge_index.size(1),1,dtype=torch.int8)
            else:
                deg = degree(data.edge_index[0],data.num_nodes,dtype=torch.int32)
                data.edge_attr = deg[data.edge_index].transpose(1,0)
            # data.edge_attr = torch.empty(data.edge_index.size(1),dtype=torch.int8) # since we only care about the size.
        if self.is_multilabel:
            if data.y.ndim > 1:
                data.y = data.y.squeeze()
            assert data.y.ndim == 1
            data.y = data.y.long()
        elif data.y.ndim == 1:
            data.y = data.y.unsqueeze(-1).float()
        
        # TODO: find a better way to handle V, maybe.
        # if data.is_directed:
        #     data.edge_index,data.edge_attr = to_undirected(data.edge_index,data.edge_attr,num_nodes=data.num_nodes)

        return data
# class Subgraph(Data):
#     node_indicator: Tensor
#     subgraph_indicator: Tensor
#     original_num_nodes: int

#     def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if key == 'node_indicator':
#             return self.original_num_nodes
#         elif key == 'subgraph_indicator':
#             return 1
#         return super().__inc__(key, value, *args, **kwargs)

# class PreprocessTransform_Ego(BaseTransform):
#     def __init__(self, max_hops: int) -> None:
#         super().__init__()
#         self.max_hops = max_hops

#     def __call__(self, data_: Data) -> MultiScaleData_2:
#         data = MultiScaleData_2()
#         data.__dict__.update(data_.__dict__)
        
#         edge_index : Tensor = data.edge_index
#         num_nodes : int = data.num_nodes

#         # first, we compute maps between edges and nodes.
#         # NOTE: we assume graph is simple and undirected.
#         edge_mask = edge_index[0] < edge_index[1]
#         inc_edge_index = edge_index[:,edge_mask]
#         data.edge_attr = data.edge_attr
#         del data.edge_index
#         data.edge_attr = data.edge_attr[edge_mask]

#         num_edges = len(data.edge_attr)
#         edges = atomspack1(inc_edge_index.transpose(1,0).flatten(),torch.arange(inc_edge_index.size(1)).repeat_interleave(2),inc_edge_index.size(1))
#         data.node2edge_index = torch.stack([
#             edges.atoms,
#             edges.domain_indicator
#         ])
#         data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)
#         # getting cycle related maps
#         cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes),self.max_cycle_size)
#         data.num_cycles = len(cycles)
        
#         cycles = [c.to_list() for c in cycles]
#         cycles = atomspack1.from_list(cycles)
#         edge2cycle : TransferData1 = TransferData1.from_atomspacks(edges,cycles,True)
#         data.set_edge2cycle_4(edge2cycle)

#         # added for debug:
#         data.num_nodes = len(data.x)
#         return data

import networkx as nx
from torch_geometric.utils import to_networkx
class PreprocessTransform(BaseTransform):
    def __init__(self, max_cycle_size: Union[int,float] = math.inf, include_cycle_map: bool = False) -> None:
        super().__init__()
        self.max_cycle_size = max_cycle_size
        self.include_cycle_map = include_cycle_map

    def __call__(self, data_: Data) -> MultiScaleData_2:
        data = MultiScaleData_2()
        data.__dict__.update(data_.__dict__)
        # assert not data_.is_directed()
        edge_index : Tensor = data.edge_index
        num_nodes : int = data.num_nodes

        # first, we compute maps between edges and nodes.
        # NOTE: we assume graph is simple and undirected.
        edge_mask = edge_index[0] < edge_index[1]
        inc_edge_index = edge_index[:,edge_mask]
        data.edge_attr = data.edge_attr
        data.edge_attr = data.edge_attr[edge_mask]

        num_edges = len(data.edge_attr)
        edges = atomspack1(inc_edge_index.transpose(1,0).flatten(),torch.arange(inc_edge_index.size(1)).repeat_interleave(2),inc_edge_index.size(1))
        data.node2edge_index = torch.stack([
            edges.atoms,
            edges.domain_indicator
        ])
        data.edge_batch = torch.zeros(num_edges,dtype=torch.int64)
        # getting cycle related maps
        cycles = list(nx.simple_cycles(to_networkx(data_,to_undirected=True),self.max_cycle_size if self.max_cycle_size != math.inf else None))#type: ignore
        del data.edge_index
        # cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes),self.max_cycle_size)
        # cycles = get_induced_cycles(from_edge_index(edge_index,num_nodes),self.max_cycle_size)
        data.num_cycles = len(cycles)
        
        # cycles = [c.to_list() for c in cycles]
        cycles = atomspack1.from_list(cycles)
        edge2cycle : TransferData1 = TransferData1.from_atomspacks(edges,cycles,True)
        data.set_edge2cycle_4(edge2cycle)

        if self.include_cycle_map:
            cycle2cycle : TransferData1 = TransferData1.from_atomspacks(cycles,cycles,False)
            data.set_cycle2cycle(cycle2cycle)
        # added for debug:
        data.num_nodes = len(data.x)
        return data

def get_pre_transform(ds: dataset_type,use_old: bool = False,max_cycle_size: Union[int,float] = math.inf, include_cycles2cycles: bool = False, store_device: str = 'cuda'):
    assert not use_old or not include_cycles2cycles
    if ds == 'graphproperty':
        tfs : list[BaseTransform] = [
            GraphPropertyPreprocessor(),
            PreprocessTransform_old(max_cycle_size) if use_old else 
            PreprocessTransform(max_cycle_size,include_cycles2cycles),
        ]
    else:
            
        tfs : list[BaseTransform] = [
            PreprocessTransform_old(max_cycle_size) if use_old else 
            PreprocessTransform(max_cycle_size,include_cycles2cycles),
        ]
        if ds in _tu_datasets:
            tfs.insert(0,TUDatasetPreprocessor(ds))
        elif ds == 'ogbg-molhiv':
            tfs.append(HIVPreprocessor())
        if ds in ['ZINC','ZINC-Full']:
            tfs.append(ZINCPreprocessor())
    return Compose(tfs)

def get_transform(ds: dataset_type, target: Literal[0,1,2,None] = None):
    if ds == 'graphproperty':
        assert target is not None
        return GraphPropertyProcessor(target)
    else:
        return Identity()