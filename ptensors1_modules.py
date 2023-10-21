import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential, Parameter
from objects import TransferData1
from ptensors1 import transfer1_1
from typing import Union

class AffineTransfer1_1(Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False,
                 intersect_reduce: str = 'sum', 
                 domain_reduce: str = 'mean', 
                 domain_transfer_reduce: str = 'sum', 
                 intersect_transfer_reduce: str = 'sum') -> None:
        r"""
        NOTE: this also considers self linear maps.
        """
        super().__init__()
        self.intersect_reduce = intersect_reduce
        self.domain_reduce = domain_reduce
        self.domain_transfer_reduce = domain_transfer_reduce
        self.intersect_transfer_reduce = intersect_transfer_reduce

        self.tf_intersect = Linear(in_channels*3,out_channels,False)
        self.tf_invariant = Linear(in_channels*2,out_channels,bias)
        
        self.linmaps_invariant = Linear(in_channels,out_channels,False)
        self.linmaps_id = Linear(in_channels,out_channels,False)
    def forward(self, x: Tensor, data: TransferData1) -> Tensor:
        (y_int, y_inv), x_inv = transfer1_1(x,data,
            self.intersect_reduce,
            self.domain_reduce,
            self.domain_transfer_reduce,
            self.intersect_transfer_reduce,
            False,
            True
            )
        inv_maps = self.linmaps_invariant(x_inv) + self.tf_invariant(y_inv)
        irred_maps = self.linmaps_id(x) + self.tf_intersect(y_int)
        
        return irred_maps + inv_maps[data.target.domain_indicator]

class LinearTransfer1_1_simple(Module):
    def __init__(self,
                 intersect_reduce: str = 'sum', 
                 domain_reduce: str = 'mean', 
                 domain_transfer_reduce: str = 'sum', 
                 intersect_transfer_reduce: str = 'sum') -> None:
        r"""
        NOTE: this also considers self linear maps.
        """
        super().__init__()
        self.intersect_reduce = intersect_reduce
        self.domain_reduce = domain_reduce
        self.domain_transfer_reduce = domain_transfer_reduce
        self.intersect_transfer_reduce = intersect_transfer_reduce

        self.epsilon_rep1 = Parameter(torch.ones(4,requires_grad=True))
        self.epsilon_rep0 = Parameter(torch.ones(3,requires_grad=True))
        
    def forward(self, x: Tensor, data: TransferData1) -> Tensor:
        (y_int, y_inv), x_inv = transfer1_1(x,data,
            self.intersect_reduce,
            self.domain_reduce,
            self.domain_transfer_reduce,
            self.intersect_transfer_reduce,
            False,
            True
            )
        # x1.view(N,k,nc).transpose(2,1).flatten(0,1).mv(w).view(N,nc)
        nc = x.size(-1)
        rep1 = torch.cat([y_int,x],-1)
        rep0 = torch.cat([y_inv,x_inv],-1)
        
        rep1 = rep1.view(-1,4,nc).transpose(2,1).flatten(0,1).mv(self.epsilon_rep1).view(-1,nc)
        rep0 = rep0.view(-1,3,nc).transpose(2,1).flatten(0,1).mv(self.epsilon_rep0).view(-1,nc)
        
        return rep1 + rep0[data.target.domain_indicator]