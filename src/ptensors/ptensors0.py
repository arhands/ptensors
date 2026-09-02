from __future__ import annotations
from torch import Tensor
from torch_scatter import scatter
from .objects1 import TransferData0

def transfer0_0(x: Tensor, transfer_data: TransferData0, reduce: str = 'sum') -> Tensor:
    messages = x[transfer_data.domain_map_edge_index[0]]
    return scatter(messages,transfer_data.domain_map_edge_index[1],0,dim_size=transfer_data.target.num_domains,reduce=reduce)
