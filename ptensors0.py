from __future__ import annotations
from torch import Tensor
from typing import Callable, Optional
from torch_scatter import scatter
from objects import TransferData0

def transfer0_0(x: Tensor, transfer_data: TransferData0, message_encoder: Optional[Callable[[Tensor,Tensor],Tensor]] = None, y: Optional[Tensor] = None, reduce: str = 'sum') -> Tensor:
    messages = x[transfer_data.domain_map_edge_index[0]]
    if message_encoder is not None:
        if y is None:
            x = y
        messages = message_encoder(messages,y[transfer_data.domain_map_edge_index[1]])
    return scatter(messages,transfer_data.domain_map_edge_index[1],0,dim_size=transfer_data.target.num_domains,reduce=reduce)
