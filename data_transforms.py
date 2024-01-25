from __future__ import annotations
from typing import Any, Literal, NamedTuple, Optional, overload
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx
from torch import Tensor
import networkx as nx
from data import FancyDataObject, supported_types
from objects import atomspack1, TransferData0, TransferData1
from objects2 import atomspack2, TransferData2

class PtensObjects:
  ap1: dict[str,atomspack1]
  ap2: dict[str,atomspack2]
  tf0: dict[tuple[str,str],TransferData0]
  tf1: dict[tuple[str,str],TransferData1]
  tf2: dict[tuple[str,str],TransferData2]
  def __init__(self,ap1,ap2,tf0,tf1,tf2) -> None:
    self.ap1 = ap1
    self.ap2 = ap2
    self.tf0 = tf0
    self.tf1 = tf1
    self.tf2 = tf2
  @overload
  def get_atomspack(self, key: str, min_order: Literal[0,1]) -> atomspack1:...
  @overload
  def get_atomspack(self, key: str, min_order: Literal[2]) -> atomspack2:...
  def get_atomspack(self, key: str, min_order: Literal[0,1,2]) -> atomspack1|atomspack2:
    if min_order <= 1 and key in self.ap1:
        return self.ap1[key]
    else:
      return self.ap2[key]

  @overload
  def get_transferData(self, key: tuple[str,str], min_order: Literal[0]) -> TransferData0:...
  @overload
  def get_transferData(self, key: tuple[str,str], min_order: Literal[1]) -> TransferData1:...
  @overload
  def get_transferData(self, key: tuple[str,str], min_order: Literal[2]) -> TransferData2:...
  def get_transferData(self, key: tuple[str,str], min_order: Literal[0,1,2]) -> TransferData0|TransferData1|TransferData2:
    if min_order == 0 and key in self.tf0:
      return self.tf0[key]
    elif min_order <= 1 and key in self.tf1:
      return self.tf1[key]
    else:
      return self.tf2[key]
  
  # def export_to_data(self, data: FancyDataObject) -> FancyDataObject:
  def export_to_data(self, data: FancyDataObject) -> None:
    ap: dict[str,atomspack1]|dict[str,atomspack2]
    for ap in [self.ap1,self.ap2]:
      key:str
      for key in ap:
        data.set_atomspack(ap[key],key)
    tf: dict[tuple[str,str],TransferData0]|dict[tuple[str,str],TransferData1]|dict[tuple[str,str],TransferData2]
    for tf in [self.tf0,self.tf1,self.tf2]:
      key2:tuple[str,str]
      for key2 in tf:
        data.set_transfer_maps(*key2,tf[key2])#type: ignore
    # return data
  @classmethod
  def from_fancy_data(cls, data: FancyDataObject) -> PtensObjects:
    return cls(*data.get_ptens_params())

  @overload
  def __getitem__(self, keyOrd: tuple[str,Literal[0,1]]) -> atomspack1:...
  @overload
  def __getitem__(self, keyOrd: tuple[str,Literal[2]]) -> atomspack2:...
  @overload
  def __getitem__(self, keyOrd: tuple[tuple[str,str],Literal[0]]) -> TransferData0:...
  @overload
  def __getitem__(self, keyOrd: tuple[tuple[str,str],Literal[1]]) -> TransferData1:...
  @overload
  def __getitem__(self, keyOrd: tuple[tuple[str,str],Literal[2]]) -> TransferData2:...
  def __getitem__(self, keyOrd: tuple[str|tuple[str,str],Literal[0,1,2]]) -> supported_types:
    key, min_order = keyOrd
    # note: this assumes the item exists and has a compatible order
    if isinstance(key,str):
      return self.get_atomspack(key,min_order)
    else:
      return self.get_transferData(key,min_order)

  @overload
  def __setitem__(self, key: str, value: atomspack1|atomspack2):...
  @overload
  def __setitem__(self, key: tuple[str,str], value: TransferData0|TransferData1|TransferData2):...
  def __setitem__(self, key: str|tuple[str,str], value: supported_types):
    if isinstance(key,str):
      if isinstance(value,atomspack2):
        self.ap2[key] = value
      else:
        self.ap1[key] = value#type: ignore
    elif isinstance(value,TransferData2):
      self.tf2[key] = value
    elif isinstance(value,TransferData1):
      self.tf1[key] = value
    else:
      self.tf0[key] = value#type: ignore

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
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    return list(data.edge_index.transpose(1,0))
class AddNodes(AddAtomspack):
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    return list(torch.arange(data.num_nodes))#type: ignore
class AddChordlessCycles(AddAtomspack):
  max_size: Optional[int]
  def __init__(self, order: Literal[1, 2], name: str, max_size: Optional[int] = None) -> None:
    super().__init__(order, name)
    self.max_size = max_size
  def get_domains(self, data: FancyDataObject) -> list[Tensor]:
    G: nx.Graph = to_networkx(data)
    cycles: list[list[int]] = nx.chordless_cycles(G,self.max_size)#type: ignore
    return [torch.tensor(c) for c in cycles]
