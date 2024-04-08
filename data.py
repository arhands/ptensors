from __future__ import annotations
import re
from typing import Any, Literal, Optional, Union, overload
import typing

from torch import Tensor
import torch
from torch_geometric.data import Data
from objects1 import atomspack1, TransferData0, TransferData1
from objects2 import atomspack2, TransferData2
supported_types = Union[atomspack1,atomspack2,TransferData0,TransferData1,TransferData2]
_prefixes = {
  'ap1',
  'ap2',
  'tf0',
  'tf1',
  'tf2',
}
def _object_to_prefix(p:supported_types) -> str:
  return {
    'atomspack1'    : "ap1",
    'atomspack2'    : "ap2",
    'TransferData0' : "tf0",
    'TransferData1' : "tf1",
    'TransferData2' : "tf2",
  }[p.__class__.__name__]


_key_regex: re.Pattern[str] = re.compile("^(" + "|".join(_prefixes) + ")__((_?[0-9a-zA-Z])+)__([0-9a-zA-Z_]+)$")
class PtensObjects:
  """
  A container class for storing ptens objects for easy access.
  """
  ap1: dict[str,atomspack1]
  ap2: dict[str,atomspack2]
  tf0: dict[tuple[str,str],TransferData0]
  """Transfer0 maps in the order they were encoded (source,dest)."""
  tf1: dict[tuple[str,str],TransferData1]
  """Transfer1 maps in the order they were encoded (source,dest)."""
  tf2: dict[tuple[str,str],TransferData2]
  """Transfer2 maps in the order they were encoded (source,dest)."""
  def __init__(self,ap1,ap2,tf0,tf1,tf2) -> None:
    self.ap1 = ap1
    self.ap2 = ap2
    self.tf0 = tf0
    self.tf1 = tf1
    self.tf2 = tf2
  @overload
  def get_atomspack(self, key: str, min_order: Literal[0,1]) -> atomspack1:...
  """Returns the atomspack1 using the specified key."""
  @overload
  def get_atomspack(self, key: str, min_order: Literal[2]) -> atomspack2:...
  def get_atomspack(self, key: str, min_order: Literal[0,1,2]) -> atomspack1|atomspack2:
    if min_order <= 1 and key in self.ap1:
        return self.ap1[key]
    else:
      return self.ap2[key]
  @overload
  def get_transferData(self, source: str, dest: str, min_order: Literal[0]) -> TransferData0:...
  @overload
  def get_transferData(self, source: str, dest: str, min_order: Literal[1]) -> TransferData1:...
  @overload
  def get_transferData(self, source: str, dest: str, min_order: Literal[2]) -> TransferData2:...
  def get_transferData(self, source: str, dest: str, min_order: Literal[0,1,2]) -> TransferData0|TransferData1|TransferData2:
    """
    Do note that source and dest are expected to be given in the order they were computed in.
    Do get the reverse, reverse the resulting object once it's returned.
    """
    key: tuple[str, str] = (source,dest)
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
    dupe = self.from_fancy_data(data)
    for c in [self.tf0,self.tf1,self.tf2,self.ap1,self.ap2]:
      for k in c:
        assert dupe[k] == self[k], f"key = {k}:\n{dupe[k]} !=\n{self[k]}"
  @classmethod
  def from_fancy_data(cls, data: FancyDataObject) -> PtensObjects:
    return cls(*data.get_ptens_params())

  @overload
  def __getitem__(self, args: str|tuple[str,Literal[0,1]]) -> atomspack1:...
  @overload
  def __getitem__(self, args: tuple[str,Literal[2]]) -> atomspack2:...
  @overload
  def __getitem__(self, args: tuple[str,str]|tuple[str,str,Literal[0]]) -> TransferData0:...
  @overload
  def __getitem__(self, args: tuple[str,str,Literal[1]]) -> TransferData1:...
  @overload
  def __getitem__(self, args: tuple[str,str,Literal[2]]) -> TransferData2:...
  def __getitem__(self, args: str|tuple[str,Literal[0,1,2]]|tuple[str,str]|tuple[str,str,Literal[0,1,2]]) -> supported_types:
    if isinstance(args,str):
      return self.get_atomspack(args,0)
    elif isinstance(args[1],int):
      args = typing.cast(tuple[str,Literal[0,1,2]],args)
      return self.get_atomspack(args[0],args[1])
    elif len(args) == 2:
      args = typing.cast(tuple[str,str],args)
      return self.get_transferData(args[0],args[1],0)
    else: # len(args) == 3
      args = typing.cast(tuple[str,str,Literal[0,1,2]],args)
      return self.get_transferData(args[0],args[1],args[2])


  @overload
  def __setitem__(self, key: str, value: atomspack1|atomspack2):...
  @overload
  def __setitem__(self, key: tuple[str,str], value: TransferData0|TransferData1|TransferData2):...
  def __setitem__(self, key: str|tuple[str,str], value: supported_types):
    if isinstance(key,str):
      if isinstance(value,atomspack2):
        self.ap2[key] = value
      else:
        value = typing.cast(atomspack1,value)
        self.ap1[key] = value
    elif isinstance(value,TransferData2):
      self.tf2[key] = value
    elif isinstance(value,TransferData1):
      self.tf1[key] = value
    else:
      value = typing.cast(TransferData0,value)
      self.tf0[key] = value
  
# def _atomspack1_inc(data: Data, prefix: str, prop_name: str, key: str, value: Union[Tensor,int]):
def _atomspack1_inc(data: Data, prop_name: str, value: Tensor|int) -> int|None:
  return {
    "atoms" : lambda: data.num_nodes,
    "domain_indicator" : lambda: value.max().item() + 1 if len(value) > 0 else 0,#type: ignore
    # "domain_indicator" : lambda: getattr(data,f"{prefix}__num_domains__{key}"),
    "num_domains" : lambda: 0,
  }[prop_name]()

def _atomspack2_inc(data: Data, prefix: str, prop_name: str, key: str, value: Tensor|int) -> int | None:
  known = {
    'col_indicator' : lambda: len(getattr(data,f"{prefix}__atoms__{key}")),
    'diag_idx' : lambda: len(getattr(data,f"{prefix}__col_indicator__{key}")),
    'row_indicator' : lambda: len(getattr(data,f"{prefix}__atoms__{key}")),
    'transpose_indicator' : lambda: len(getattr(data,f"{prefix}__col_indicator__{key}")),
  }
  if prop_name in known:
    return known[prop_name]()
  else:
    return _atomspack1_inc(data,prop_name,value)

def _transfer_map_to_atomspacks_attr(data: FancyDataObject, key: str, attr: str, prefix: Literal[None,'ap1','ap2'] = None) -> tuple[Tensor, Tensor]:
  options: list[Literal['ap1','ap2']] = ['ap1','ap2']
  if prefix is None:
    checking: list[Literal['ap1','ap2']] = options
  else:
    checking = [prefix]
  [source, target] = key.split('_and_')
  for _prefix in checking:
    if hasattr(data,f'{_prefix}__{attr}__{source}'):
      # NOTE: we assume source and target will always exist with the same order.
      return getattr(data,f'{_prefix}__{attr}__{source}'), getattr(data,f'{_prefix}__{attr}__{target}')
  raise Exception




def _transferData0_inc(data: FancyDataObject, prefix: str, key: str) -> Tensor:
  # the only prop this has is 'domain_map_edge_index'
  source, dest = _transfer_map_to_atomspacks_attr(data,key,'num_domains')
  if isinstance(source,Tensor):
    source = source.sum().item()
    dest = dest.sum().item()
  return torch.tensor([source,dest],device=data.x.device).unsqueeze(-1)

def _transferData1_inc(data: FancyDataObject, prefix: str, prop_name: str, key: str):
  def eval_node_map_edge_index():
    source_atoms, target_atoms = _transfer_map_to_atomspacks_attr(data,key,'atoms')
    return torch.tensor([
      [len(source_atoms)],
      [len(target_atoms)],
    ],device=data.x.device)
  known = {
    'intersect_indicator' : lambda : getattr(data,f"{prefix}__domain_map_edge_index__{key}").size(1),
    'node_map_edge_index' : eval_node_map_edge_index,
    # 'num_nodes' : eval_node_map_edge_index,
  }
  if prop_name in known:
    return known[prop_name]()
  else:
    return _transferData0_inc(data,prefix,key)

def _transferData2_inc(data: FancyDataObject, prefix: str, prop_name: str, key: str):
  # a = TransferData2()
  # a.ij_indicator
  # a.node_pair_map
  def eval_node_pair_map_edge_index():
    source_atoms, target_atoms = _transfer_map_to_atomspacks_attr(data,key,'row_indicator','ap2')
    return torch.tensor([
      [len(source_atoms)],
      [len(target_atoms)],
    ],device=data.x.device)
  known = {
    'ij_indicator'  : lambda : getattr(data,f"{prefix}__node_map_edge_index__{key}").size(1),
    'node_pair_map' : eval_node_pair_map_edge_index,
  }
  if prop_name in known:
    return known[prop_name]()
  else:
    return _transferData1_inc(data,prefix,prop_name,key)



class FancyDataObject(Data):
  """
    Handles the batching of ptens-type objects once they have been pre-processed.
  """
  def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
    parse: re.Match[str] | None  = _key_regex.match(key)
    if parse is not None:
      prefix, prop_name, _, key = parse.groups()
      return {
        'tf0' : lambda : _transferData0_inc(self,prefix,key),
        'tf1' : lambda : _transferData1_inc(self,prefix,prop_name,key),
        'tf2' : lambda : _transferData2_inc(self,prefix,prop_name,key),
        'ap1' : lambda : _atomspack1_inc(self,prop_name,value),
        'ap2' : lambda : _atomspack2_inc(self,prefix,prop_name,key,value),
      }[prefix]()
    return super().__inc__(key, value, *args, **kwargs)
  def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
    if key[:len("tf2__node_pair_map")] == 'tf2__node_pair_map':
      return 1
    else:
      return super().__cat_dim__(key, value, *args, **kwargs)

  def _set_ptens_param(self, key: str, param: supported_types, ignore_keys: list[str]|set[str]):
    """NOTE: will ignore atomspacks, please handle them separately"""
    # assert "__" not in key
    #NOTE: we require that properties in ptens objects do not contain double underscores.
    prefix: str = _object_to_prefix(param)
    dictionary: dict[str, Any] = vars(param)
    for k in dictionary:
      if k not in ignore_keys:
        setattr(self,f"{prefix}__{k}__{key}",dictionary[k])
        # assert hasattr(self,f"{prefix}__{k}__{key}")
  def get_ptens_params(self) -> tuple[dict[str, atomspack1], dict[str, atomspack2], dict[tuple[str, str], TransferData0], dict[tuple[str, str], TransferData1], dict[tuple[str, str], TransferData2]]:
    # dictionary = vars(self)
    args: dict[str,Any] = dict()
    transfer_objects: list[list[tuple[str,str,str]]] = [[],[],[]]
    atomspack_objects: list[list[str]] = [[],[]]
    for k in self.keys:#type: ignore
      res: re.Match[str] | None = _key_regex.match(k)
      if res is not None:
        prefix, prop_name, _, key = res.groups()
        if key in args:
          args[key][prop_name] = getattr(self,k)
        else:
          args[key] = { prop_name : getattr(self,k)}
        if prefix[:2] == 'ap':
          atomspack_objects[int(prefix[-1]) - 1].append(key)
        elif prefix[:2] == 'tf':
          transfer_objects[int(prefix[-1])].append((*key.split("_and_"),key))#type: ignore
        else:
          raise Exception(prefix)
    #
    ap1: dict[str,atomspack1] = {key: atomspack1(**args[key]) for key in atomspack_objects[0]}
    ap2: dict[str,atomspack2] = {key: atomspack2(**args[key]) for key in atomspack_objects[1]}
    def tf_to_ap(source: str, dest: str) -> tuple[atomspack1, atomspack1] | tuple[atomspack2, atomspack2]:
      if source in ap1:
        return ap1[source], ap1[dest]
      else:
        return ap2[source], ap2[dest]
    def tf_to_ap2(source: str, dest: str) -> tuple[atomspack2, atomspack2]:
      return ap2[source], ap2[dest]
    tf0: dict[tuple[str,str],TransferData0] = {(source,dest): TransferData0(*tf_to_ap(source,dest),**args[key]) for source,dest,key in transfer_objects[0]}
    tf1: dict[tuple[str,str],TransferData1] = {(source,dest): TransferData1(*tf_to_ap(source,dest),num_nodes=self.num_nodes,**args[key]) for source,dest,key in transfer_objects[1]}
    tf2: dict[tuple[str,str],TransferData2] = {(source,dest): TransferData2(*tf_to_ap2(source,dest),num_nodes=self.num_nodes,**args[key]) for source,dest,key in transfer_objects[2]}
    return ap1, ap2, tf0, tf1, tf2
  def set_atomspack(self, ap1: atomspack1|atomspack2, key: str) -> None:
    self._set_ptens_param(key,ap1,['_atoms2','_domains_indicator2','raw_num_domains'])
  def set_transfer_maps(self, source_key: str, target_key: str, value: TransferData0|TransferData1|TransferData2) -> None:
    """NOTE: you are expected to separately set the source and target atomspacks."""
    # assert "_and_" not in source_key
    # assert "_and_" not in target_key
    self._set_ptens_param(f'{source_key}_and_{target_key}',value,['source','num_nodes','target','_atoms2','_domains_indicator2'])
  @staticmethod
  def assert_valid_domain_name(name: str) -> None:
    """
      A helper method for checking if a name is valid. 
      This is not called here, but is useful for transforms and
      such for input validation when they are created.
    """
    assert "__" not in name
    assert "_and_" not in name
    assert re.match("^[_0-9a-zA-Z]+$",name) is not None