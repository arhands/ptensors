from __future__ import annotations
from typing import Callable, Literal, NamedTuple, Optional, Union
import torch
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.data import Data
from objects import TransferData1, atomspack1, atomspack2
from ptensors1 import linmaps0_1, linmaps1_0, transfer0_1, transfer1_0, transfer1_1
from ptensors0 import transfer0_0

class Ptensors(NamedTuple):
    values: Tensor
    domains: atomspack1
    order: Literal[0,1]
    def __add__(self, other: Union[Tensor,Ptensors,Ptensors0_1]):
        if isinstance(other,Ptensors):
            assert other.order == self.order
            v = self.values + other.values # we assume domains are correct...
        elif isinstance(other,Ptensors0_1):
            return self + other._manifest()
        else:
            v = self.values + other
        return Ptensors(v,self.domains,self.order)
    
    def linmaps0(self, reduce: str = 'sum') -> Ptensors:
        assert self.order == 1
        return Ptensors(scatter(self.values,self.domains.domain_indicator,0,reduce=reduce),self.domains,0)
    
    def linmaps1(self, reduce: str = 'sum') -> Union[CatPtensors1,Ptensors0_1]:
        if self.order == 0:
            return Ptensors0_1(self.values,self.domains)
        else: # order is 1.
            return CatPtensors1([
                Ptensors0_1(linmaps1_0(self.values,self.domains,reduce),self.domains),
                self
            ])
    
    def transfer0(self, tmap: TransferData1, reduce: Union[str,list[str]] = 'sum'):
        if self.order == 0:
            assert isinstance(reduce,str)
            v = transfer0_0(self.values,tmap,reduce)
        else: # order is 1
            v = transfer1_0(self.values,tmap,reduce)
        return Ptensors(v,tmap.target,0)
    
    def transfer1(self, tmap: TransferData1, reduce: Union[str,list[str]] = 'sum') -> CatPtensors1:
        if self.order == 0:
            reps = [
                
            ]
            return Ptensors0_1(self.values,self.domains)
        else: # order is 1.
            return CatPtensors1(
                Ptensors0_1(linmaps1_0(self.values,self.domains,reduce),self.domains),
                self
            )

# class SplitLinear1(NamedTuple):
#     lower_linear: Linear
#     linear: Linear

class Ptensors0_1(NamedTuple):
    r"""A ptensors1 represented by a ptensors0."""
    values: Tensor
    domains: atomspack1
    def _manifest(self) -> Ptensors:
        return Ptensors(self.values[self.domains.domain_indicator],self.domains,1)
    def __add__(self, other):
        if isinstance(other,Ptensors):
            return self._manifest() + other


class CatPtensors1(NamedTuple):
    values: list[Union[Ptensors0_1,Ptensors]]

class SplitPtensors1:
    values: Tensor
    domains: atomspack1
    lower_values: Union[None,Tensor]
    to_upper_reduction: Union[None,str]
    r"""In case part of the tensor can be represented as a ptensors0."""
    def __init__(self,domains: atomspack1, values: Union[None,Tensor], lower_values: Union[None,Tensor], to_upper_reduction: Union[str,None]) -> None:
        self.values = values
        self.domains = domains
        self.lower_values = lower_values
        self.to_upper_reduction = to_upper_reduction