from __future__ import annotations
from typing import Any, NamedTuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from objects import atomspack1, TransferData1, TransferData0, MultiTransferData1

class Data1(Data):
    r"""
    
    """
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        return super().__cat_dim__(key, value, *args, **kwargs)
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'atomspack_' == key[:10]:
            key = key[10:]
            if key[-5:] == 'atoms':
                return self.num_nodes
            elif key[-16:] == 'domain_indicator':
                identifier = key[:-17]
                return getattr(self,f"atomspack_{identifier}_num_domains")
            else:
                raise NameError(f'key "{key}" is forbidden.')
        return super().__inc__(key, value, *args, **kwargs)
#################################################################################################################################
# functions
#################################################################################################################################

def add_atomspack1(data: Data1, pack: atomspack1, identifier: str) -> Data1:
    identifier = f"atomspack_{identifier}"
    setattr(data,f"{identifier}_atoms",pack.atoms)
    setattr(data,f"{identifier}_num_domains",pack.num_domains)
    setattr(data,f"{identifier}_domain_indicator",pack.domain_indicator)


def add_multi_transfer_data1(data: Data1, transfer_data: MultiTransferData1, identifier: str):
    for idx, atomspack in enumerate(transfer_data.atomspacks):
        add_atomspack1(data,atomspack,f'{identifier}_{idx}')
    identifier = f"transfer_data_{identifier}"
    for key in transfer_data.transfer_data:
        if key[0] <= key[1]:
            base_name = f"{identifier}_{idx[0]}_{idx[1]}"
            value = transfer_data[key]
            setattr(data,f"{base_name}_domain_map_edge_index",value.domain_map_edge_index)
            setattr(data,f"{base_name}_node_map_edge_index",value.node_map_edge_index)
            setattr(data,f"{base_name}_intersect_indicator",value.intersect_indicator)
            setattr(data,f"{base_name}_num_intersections",value.intersect_indicator.max() + 1)