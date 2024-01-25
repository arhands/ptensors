from __future__ import annotations
from typing import Any, Literal, NamedTuple, Optional, overload
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx
from torch import Tensor
import networkx as nx
from data import FancyDataObject, supported_types, PtensObjects
from objects import atomspack1, TransferData0, TransferData1
from objects2 import atomspack2, TransferData2

class InitPtensData(BaseTransform):
  def __call__(self, data: Data) -> tuple[FancyDataObject,PtensObjects]:
    data2 = FancyDataObject()
    data2.__dict__.update(data.__dict__)
    return data2, PtensObjects(dict(),dict(),dict(),dict(),dict())
class AddAtomspack(BaseTransform):
  order_idx: Literal[0,1]
  name: str
  def __init__(self, order: Literal[1,2], name: str) -> None:
    super().__init__()
    assert "__" not in name
    self.order_idx = order - 1
    self.name = name
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:...

  def __call__(self, data: tuple[FancyDataObject,PtensObjects]) -> tuple[FancyDataObject,PtensObjects]:
    domains: list[Tensor] = self.get_domains(data[0])
    data[1][self.name] = [
      atomspack1,
      atomspack2
    ][self.order_idx].from_tensor_list(domains)
    return data

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
  def __call__(self, data: tuple[FancyDataObject,PtensObjects]) -> tuple[FancyDataObject,PtensObjects]:
    pdata: PtensObjects = data[1]
    pdata[(self.source,self.target)] = \
    [
      TransferData0,
      TransferData1,
      TransferData2,
    ][self.order].from_atomspacks(pdata.get_atomspack(self.source,self.order),pdata.get_atomspack(self.target,self.order),self.ensure_subset)
    return data

class FinalizePtensData(BaseTransform):
  def __call__(self, data: tuple[FancyDataObject,PtensObjects]) -> FancyDataObject:
    G, pdata = data
    pdata.export_to_data(G)
    # TODO: make sure I can actually do what the below line is doing.
    G.edge_index = None#type: ignore
    return data[0]

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
      edge_index = edge_index[:,edge_index[1]>=edge_index[0]]
    return list(edge_index.transpose(1,0))

class AddNodes(AddAtomspack):
  def __init__(self, order: Literal[1, 2], name: str = 'nodes') -> None:
    super().__init__(order, name)
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    return list(torch.arange(data.num_nodes))#type: ignore

class AddChordlessCycles(AddAtomspack):
  max_size: Optional[int]
  def __init__(self, order: Literal[1, 2], max_size: Optional[int] = None, name: str = 'cycles') -> None:
    super().__init__(order, name)
    self.max_size = max_size
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    G: nx.Graph = to_networkx(data)
    cycles: list[list[int]] = nx.chordless_cycles(G,self.max_size)#type: ignore
    return [torch.tensor(c) for c in cycles]
