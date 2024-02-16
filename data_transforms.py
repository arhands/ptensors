from __future__ import annotations
from math import inf
from typing import Any, Literal, NamedTuple, Optional, overload
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, degree
from torch import Tensor
# import networkx as nx
from data import FancyDataObject, supported_types, PtensObjects
from objects1 import atomspack1, TransferData0, TransferData1
from objects2 import atomspack2, TransferData2
from induced_cycle_finder import get_induced_cycles, from_edge_index

class GeneratePtensObject(BaseTransform):
  def __init__(self, atomspacks: list[AddAtomspack], transfers: list[AddTransferMap]) -> None:
    super().__init__()
    self.atomspacks: list[AddAtomspack] = atomspacks
    self.transfers: list[AddTransferMap] = transfers
  def __call__(self, data: Data) -> FancyDataObject:
    data2 = FancyDataObject()
    data2.__dict__.update(data.__dict__)
    ptobj = PtensObjects(dict(),dict(),dict(),dict(),dict())
    for ap in self.atomspacks:
      ptobj: PtensObjects = ap(data2,ptobj)
    for tf in self.transfers:
      ptobj: PtensObjects = tf(ptobj)
    ptobj.export_to_data(data2)
    # TODO: make sure I can actually do what the below line is doing.
    data2.edge_index = None#type: ignore
    return data2

class AddAtomspack:
  order_idx: Literal[0,1]
  name: str
  def __init__(self, order: Literal[1,2], name: str) -> None:
    super().__init__()
    assert "__" not in name
    self.order_idx = order - 1
    self.name = name
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:...

  def __call__(self, data: FancyDataObject, objs: PtensObjects) -> PtensObjects:
    domains: list[Tensor] = self.get_domains(data)
    objs[self.name] = [
      atomspack1,
      atomspack2
    ][self.order_idx].from_tensor_list(domains)
    return objs

class AddTransferMap(BaseTransform):
  order: Literal[0,1,2]
  source: str
  target: str
  ensure_subset: bool
  def __init__(self, source: str, target: str, order: Literal[0,1,2], ensure_subset: bool) -> None:
    super().__init__()
    assert "_and_" not in source
    assert "_and_" not in target
    self.source = source
    self.target = target
    self.order = order
    self.ensure_subset = ensure_subset
  def __call__(self, data: PtensObjects) -> PtensObjects:
    data[(self.source,self.target)] = \
    [
      TransferData0,
      TransferData1,
      TransferData2,
    ][self.order].from_atomspacks(data.get_atomspack(self.source,self.order),data.get_atomspack(self.target,self.order),self.ensure_subset)
    return data

#################################################################################################################################
# subgraph specific transforms
#################################################################################################################################

class AddEdges(AddAtomspack):
  graph_is_undirected: bool
  def __init__(self, order: Literal[1, 2], graph_is_undirected : bool = True, name: str = 'edges') -> None:
    super().__init__(order, name)
    self.graph_is_undirected = graph_is_undirected
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    edge_index: Tensor = data.edge_index
    if self.graph_is_undirected:
      mask = edge_index[1]>=edge_index[0]
      edge_index = edge_index[:,mask]
      # TODO: find a better place for the following line...
      data.edge_attr = data.edge_attr[mask]#type: ignore
    return list(edge_index.transpose(1,0))

class AddNodes(AddAtomspack):
  def __init__(self, order: Literal[1, 2], name: str = 'nodes') -> None:
    super().__init__(order, name)
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    return list(torch.arange(data.num_nodes).unsqueeze(-1))#type: ignore

class AddChordlessCycles(AddAtomspack):
  max_size: Optional[int]
  undirected: bool
  def __init__(self, order: Literal[1, 2], max_size: Optional[int] = None, undirected: bool = True, name: str = 'cycles') -> None:
    super().__init__(order, name)
    self.max_size = max_size
    self.undirected = undirected
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    # G: nx.Graph = to_networkx(data,to_undirected=self.undirected)# TODO: add check
    # cycles: list[list[int]] = nx.chordless_cycles(G,self.max_size)#type: ignore
    cycles = get_induced_cycles(from_edge_index(data.edge_index,data.num_nodes),self.max_size if self.max_size is not None else inf)#type: ignore
    return [torch.tensor(c.to_list()) for c in cycles]

#################################################################################################################################
# dataset specific transforms
#################################################################################################################################
class TUPreprocessingBase(BaseTransform):
  def __init__(self, ds: str) -> None:
    super().__init__()
    self.is_multilabel = ds in ['COLLAB','IMDB-MULTI','ENZYMES']
    self.ignore_degree = ds == 'REDDIT_BINARY'
  @overload
  def __call__(self, data: FancyDataObject) -> FancyDataObject:...
  @overload
  def __call__(self, data: Data) -> Data:...
  def __call__(self, data: Data|FancyDataObject) -> Data|FancyDataObject:
    if data.x is not None:
        data.x = data.x.argmax(1)#type: ignore
    else:
        # data.x = torch.empty(data.num_nodes,dtype=torch.int8) # since we only care about the size.
        if self.ignore_degree:
            data.x = torch.zeros(data.num_nodes,dtype=torch.int8)#type: ignore
        else:
            data.x = degree(data.edge_index[0],data.num_nodes,dtype=torch.int32)#type: ignore
    if data.edge_attr is not None:
        if data.edge_attr.ndim > 1:
            data.edge_attr = data.edge_attr.argmax(1)#type: ignore
    else:
        if self.ignore_degree:
            data.edge_attr = torch.zeros(data.edge_index.size(1),1,dtype=torch.int8)#type: ignore
        else:
            deg = degree(data.edge_index[0],data.num_nodes,dtype=torch.int32)
            data.edge_attr = deg[data.edge_index].transpose(1,0)#type: ignore
        # data.edge_attr = torch.empty(data.edge_index.size(1),dtype=torch.int8) # since we only care about the size.
    if self.is_multilabel:
        if data.y.ndim > 1:
            data.y = data.y.squeeze()#type: ignore
        assert data.y.ndim == 1
        data.y = data.y.long()#type: ignore
    elif data.y.ndim == 1:
        data.y = data.y.unsqueeze(-1).float()#type: ignore
    
    # TODO: find a better way to handle V, maybe.
    # if data.is_directed:
    #     data.edge_index,data.edge_attr = to_undirected(data.edge_index,data.edge_attr,num_nodes=data.num_nodes)

    return data