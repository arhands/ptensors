from __future__ import annotations
from typing import Literal, Optional, overload
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, degree
from torch import Tensor
from data import FancyDataObject, PtensObjects
from objects1 import atomspack1, TransferData0, TransferData1
from objects2 import atomspack2, TransferData2
import networkx as nx

class GeneratePtensObject(BaseTransform):
  def __init__(self, atomspacks: list[AddAtomspack], transfers: list[AddTransferMap], return_ptens_obj: bool = False) -> None:
    super().__init__()
    self.atomspacks: list[AddAtomspack] = atomspacks
    self.transfers: list[AddTransferMap] = transfers
    self.return_ptens_obj = return_ptens_obj
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
    if self.return_ptens_obj:
      return data2, ptobj
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
    G: nx.Graph = to_networkx(data,to_undirected=self.undirected)# TODO: add check
    return [torch.tensor(c) for c in nx.simple_cycles(G,self.max_size)]#type: ignore
    # cycles = get_induced_cycles(from_edge_index(data.edge_index,data.num_nodes),self.max_size if self.max_size is not None else inf)#type: ignore
    # return [torch.tensor(c.to_list()) for c in cycles]

#################################################################################################################################
# dataset specific transforms
#################################################################################################################################
encoding_flags = Literal['OGB','degree',None]
label_type = Literal['single-dim','multi-label','multi-class']
class StandardPreprocessing(BaseTransform):
  label: label_type
  node_encoding: encoding_flags
  edge_encoding: encoding_flags
  def __init__(self, label: label_type, node_encoding: encoding_flags, edge_encoding: encoding_flags) -> None:
    """NOTE: encoding just gives some flags for how you want it to process."""
    super().__init__()
    self.label = label
    self.node_encoding = node_encoding
    self.edge_encoding = edge_encoding
  @overload
  def __call__(self, data: FancyDataObject) -> FancyDataObject:...
  @overload
  def __call__(self, data: Data) -> Data:...
  def __call__(self, data: Data|FancyDataObject) -> Data|FancyDataObject:
    # node proc
    x: Tensor|None = data.x
    deg: None|Tensor = None
    if x is None:
      if self.node_encoding == 'degree':
        deg = degree(data.edge_index[0],data.num_nodes,dtype=torch.int32)
        x = deg
      else:
        x = torch.zeros(data.num_nodes,dtype=torch.int8)#type: ignore
    elif self.node_encoding != 'OGB':
      # we want to ensure standard one-hot form.
      if x.ndim == 2:
        if  x.size(1) > 1:
          x = x.argmax(1)
        else:
          x = x.flatten()
      else:
        x = x.long()
    data.x = x#type: ignore

    # edge proc
    edge_attr: Tensor|None = data.edge_attr
    if edge_attr is None:
      if self.edge_encoding == 'degree':
        if deg is None:
          deg = degree(data.edge_index[0],data.num_nodes,dtype=torch.int32)
        edge_attr = deg[data.edge_index].transpose(1,0)
      else:
        edge_attr = torch.zeros(data.edge_index.size(1),1,dtype=torch.int8)
    elif self.edge_encoding != 'OGB' and edge_attr.ndim == 2:
      if edge_attr.size(1) > 1:
        edge_attr = edge_attr.argmax(1)
      else:
        edge_attr = edge_attr.flatten()
    data.edge_attr = edge_attr#type: ignore

    # graph labels
    y: Tensor = data.y
    if self.label == 'multi-class':
      if y.ndim > 1:
        y = y.squeeze()
      y = y.long()
    elif self.label == 'single-dim':
      y = y.view(-1,1).float()
    data.y = y#type: ignore

    return data

# class ZINCPreProcessingBase(BaseTransform):
#   @overload
#   def __call__(self, data: FancyDataObject) -> FancyDataObject:...
#   @overload
#   def __call__(self, data: Data) -> Data:...
#   def __call__(self, data: Data|FancyDataObject) -> Data|FancyDataObject:
#     data.x = data.x.flatten()#type: ignore
#     data.y = data.y.unsqueeze(-1)#type: ignore
#     return data

# class OGBPreprocessingBase(BaseTransform):
#   @overload
#   def __call__(self, data: FancyDataObject) -> FancyDataObject:...
#   @overload
#   def __call__(self, data: Data) -> Data:...
#   def __call__(self, data: Data|FancyDataObject) -> Data|FancyDataObject:
#     data.y = data.y.float()
#     return data