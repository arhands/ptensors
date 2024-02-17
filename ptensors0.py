from __future__ import annotations
from torch import Tensor
from typing import Callable, Optional
from torch_scatter import scatter
from objects1 import TransferData0

def transfer0_0(x: Tensor, transfer_data: TransferData0, reduce: str = 'sum') -> Tensor:
    messages = x[transfer_data.domain_map_edge_index[0]]
    return scatter(messages,transfer_data.domain_map_edge_index[1],0,dim_size=transfer_data.target.num_domains,reduce=reduce)

def transfer0_0_msg(x: Tensor, transfer_data: TransferData0, message_encoder: Callable[[Tensor],Tensor], reduce: str = 'sum') -> Tensor:
    messages = x[transfer_data.domain_map_edge_index[0]]
    messages = message_encoder(messages)
    return scatter(messages,transfer_data.domain_map_edge_index[1],0,dim_size=transfer_data.target.num_domains,reduce=reduce)

def transfer0_0_bi_msg(x: Tensor, transfer_data: TransferData0, message_encoder: Callable[[Tensor,Tensor],Tensor], y: Tensor, reduce: str = 'sum') -> Tensor:
    messages = x[transfer_data.domain_map_edge_index[0]]
    messages = message_encoder(messages,y[transfer_data.domain_map_edge_index[1]])
    return scatter(messages,transfer_data.domain_map_edge_index[1],0,dim_size=transfer_data.target.num_domains,reduce=reduce)
