"""
abbreviations:
    - int - intersection between domains
    - inv - permutation invariant map
"""
from __future__ import annotations
from typing import Union
import torch
from torch import Tensor
from torch_scatter import scatter
# from torch_geometric.data import Data
from .objects1 import TransferData1, atomspack1

def linmaps0_1(x: Tensor, domains:  atomspack1):
    # print("x.size()",x.size())
    # print("domains.domain_indicator.size()",domains.domain_indicator.size())
    return x[domains.domain_indicator]

def linmaps1_0(x: Tensor, domains: atomspack1, reduce: str='sum'):
    return scatter(x,domains.domain_indicator,0,reduce=reduce)

def linmaps1_1(x: Tensor, domains: atomspack1, reduce: str='sum'):
    return torch.cat([
        x,
        linmaps0_1(linmaps1_0(x,domains,reduce),domains),
    ],-1)

def transfer0_1(x: Tensor, data: TransferData1, reduce: Union[list[str],str]='sum'):
    r"""
        for transfering from a ptensors0 to a ptensors1
        first reduce: reductions between different ptensors on specific atoms.
        second reduce: reductions between different ptensors on for incoming values (entire ptensors).
    """
    if isinstance(reduce,str):
        reduce = [reduce]*2
    to_target = x[data.domain_map_edge_index[0]]
    intersection_broadcasted = scatter(to_target[data.intersect_indicator],data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[0])
    target_broadcasted = scatter(to_target,data.domain_map_edge_index[1],0,dim_size=data.target.num_domains,reduce=reduce[1])[data.target.domain_indicator]

    return torch.cat([
        intersection_broadcasted,
        target_broadcasted,
    ],-1)

# def transfer0_1_bi_msg(x: Tensor, data: TransferData1, encoder_int: Callable[[Tensor,Tensor],Tensor], encoder_inv: Callable[[Tensor,Tensor],Tensor], y: Tensor, reduce: Union[list[str],str]='sum'):
#     r"""
#         for transfering from a ptensors0 to a ptensors1
#     """
#     if isinstance(reduce,str):
#         reduce = [reduce]*4
#     x_inv = x[data.domain_map_edge_index[0]]
#     x_int = x_inv[data.intersect_indicator]

#     y_inv = scatter(y,data.target.domain_indicator,0,dim_size=data.target.num_domains,reduce=reduce[0])
#     y_inv = y_inv[data.domain_map_edge_index[1]]
#     y_int = y[data.node_map_edge_index[1]]
#     y_int_inv = scatter(y_int,data.intersect_indicator,0,reduce=reduce[1])

#     msg_int = encoder_int(x_int,torch.cat([y_int,y_inv[data.intersect_indicator]],-1))
#     msg_inv = encoder_inv(x_inv,torch.cat([y_inv,y_int_inv],-1))

#     out_int = scatter(msg_int,data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=reduce[2])
#     out_inv = scatter(msg_inv,data.domain_map_edge_index[1],0,dim_size=data.target.num_domains,reduce=reduce[3])

#     return (
#         out_int,
#         out_inv[data.target.domain_indicator],
#     )

# def transfter0_1_0_bi_msg_raw(
#         x: Tensor, 
#         y: Tensor, 
#         data: TransferData1, 
#         x_to_x_encoder_int: Callable[[Tensor,Tensor,Tensor],Tensor], 
#         x_to_x_encoder_inv: Callable[[Tensor,Tensor],Tensor], 
#         x_to_y_encoder_int: Callable[[Tensor,Tensor,Tensor],Tensor],
#         x_to_y_encoder_inv: Callable[[Tensor,Tensor],Tensor], reduce: Union[list[str],str]) -> tuple[tuple[Tensor,Tensor],tuple[Tensor,Tensor],Tensor]:
#     r""""
#     params:
#         x: input ptensor 0
#         y: target ptensor 1
#         data: transfer data from x to y
#         x_to_x_encoder_int: update function for messages going from x to y then to x
#         x_to_x_encoder_inv: update function for messages going from x to y then to x
#         x_to_y_encoder_int: update function for messages going from x to y
#         x_to_y_encoder_inv: update function for messages going from x to y
    
#     NOTE: this is equivalent to calling 'transfer0_1_bi_msg' from x to y, then 'transfer1_0' 
#         back to x and then calling 'transfer0_1_bi_msg' from x to y again to compute messages just going to y, with the slight modification that self messages are removed.
#     """
#     # TODO: add decoders
#     # TODO: make all encoders/decoders optional
#     if isinstance(reduce,str):
#         reduce = [reduce]*3

#     x_inv = x[data.domain_map_edge_index[0]]
#     x_int = x_inv[data.intersect_indicator]

#     y_int = y[data.node_map_edge_index[1]]

#     y_inv = scatter(y,data.target.domain_indicator,0,reduce=reduce[0])
#     y_inv = y_inv[data.domain_map_edge_index[1]]
#     y_inv_int = y_inv[data.intersect_indicator]

#     # encoding messages (mapping x to y)
#     x_to_x_encoded_inv_msg = x_to_x_encoder_inv(x_inv,y_inv)
#     x_to_x_encoded_inv = scatter_sum(x_to_x_encoded_inv_msg,data.domain_map_edge_index[1],0)

#     x_to_y_encoded_inv_msg = x_to_y_encoder_inv(x_inv,y_inv)
#     y_inv_out = scatter(x_to_y_encoded_inv_msg,data.domain_map_edge_index[1],0,reduce=reduce[1])
    
#     x_to_x_encoded_int_msg = x_to_x_encoder_int(x_int,y_int,y_inv_int)
#     x_to_x_encoded_int = scatter_sum(x_to_x_encoded_int_msg,data.node_map_edge_index[1],0)

#     x_to_y_encoded_int_msg = x_to_y_encoder_int(x_int,y_int,y_inv_int)
#     y_int_out = scatter_sum(x_to_y_encoded_int_msg,data.node_map_edge_index[1],0)

#     # decoding messages (mapping back)
#     x_to_x_decoded_inv_msg = x_to_x_encoded_inv[data.domain_map_edge_index[1]]
#     x_to_x_decoded_int_msg = x_to_x_encoded_int[data.node_map_edge_index[1]]

#     # removing self messages
#     x_to_x_decoded_inv_msg = x_to_x_decoded_inv_msg - x_to_x_encoded_inv_msg
#     x_to_x_decoded_int_msg = x_to_x_decoded_int_msg - x_to_x_encoded_int_msg

#     # aggregating results and returning
#     x_int_out = scatter(x_to_x_decoded_int_msg,data.node_map_edge_index[0],0,reduce=reduce[2])
#     x_inv_out = scatter(x_to_x_decoded_inv_msg,data.domain_map_edge_index[0],0,reduce=reduce[3])

#     return (x_int_out,x_inv_out), (y_int_out,y_inv_out), y_int


# def transfer1_0_msg(x: Tensor, data: TransferData1, encoder_int: Optional[Callable[[Tensor],Tensor]], encoder_inv: Optional[Callable[[Tensor],Tensor]], reduce: Union[list[str],str]='sum'):
#     r"""for transfering from a ptensors0 to a ptensors1."""
#     if isinstance(reduce,str):
#         reduce = [reduce]*3
    
#     res = []
#     if encoder_int is not None:
#         int_msg = x[data.node_map_edge_index[0]]
#         int_msg = encoder_int(int_msg)
#         node_maps = scatter(int_msg,data.target.domain_indicator[data.node_map_edge_index[1]],0,dim_size=data.target.num_domains,reduce=reduce[2])
#         res.append(node_maps)

#     if encoder_inv is not None:
#         domain_reduced = scatter(x,data.source.domain_indicator,0,reduce=reduce[0])
#         inv_msg = domain_reduced[data.domain_map_edge_index[0]]
#         inv_msg = encoder_inv(inv_msg)
#         domain_maps = scatter(inv_msg,data.domain_map_edge_index[1],0,dim_size=data.target.num_domains,reduce=reduce[1])
#         res.append(domain_maps)
#     return torch.cat(res,-1)

def transfer1_0(x: Tensor, data: TransferData1, reduce: Union[list[str],str]='sum', return_list: bool = False):
    r"""for transfering from a ptensors0 to a ptensors1"""
    if isinstance(reduce,str):
        reduce = [reduce]*3
    
    domain_reduced = scatter(x,data.source.domain_indicator,0,reduce=reduce[0])
    domain_maps = scatter(domain_reduced[data.domain_map_edge_index[0]],data.domain_map_edge_index[1],0,dim_size=data.target.num_domains,reduce=reduce[1])

    node_maps = scatter(x[data.node_map_edge_index[0]],data.target.domain_indicator[data.node_map_edge_index[1]],0,dim_size=data.target.num_domains,reduce=reduce[2])
    # node_maps = scatter(node_maps,,0,dim_size=data.target.num_domains,reduce=reduce[3])
    ret = [
        node_maps,
        domain_maps,
    ]
    if return_list:
        return ret
    return torch.cat(ret,-1)

# def transfer1_1(x: Tensor, data: TransferData1, reduce: Union[list[str],str]='sum', combine_reps: bool = True, return_domain_reduction: bool = False):
def transfer1_1(
        x: Tensor, 
        data: TransferData1, 
        intersect_reduce: str = 'sum', 
        domain_reduce: str = 'mean', 
        domain_transfer_reduce: str = 'sum', 
        intersect_transfer_reduce: str = 'sum', 
        combine_reps: bool = True, 
        return_domain_reduction: bool = False):
    r"""
    for transfering from a ptensors1 to a ptensors1
    - args:
        + combine_reps: if set to true, will broadcast zeroth order reps to first order and return as single tensor.
        + return_domain_reduction: if set to true, will return the linmaps0 of the input tensor.
    """

    x_intersect = x[data.node_map_edge_index[0]]
    x_intersect_reduced = scatter(x_intersect,data.intersect_indicator,0,reduce=intersect_reduce)
    x_domain = scatter(x,data.source.domain_indicator,0,reduce=domain_reduce)
    x_invar = torch.cat([
        x_intersect_reduced, # local
        x_domain[data.domain_map_edge_index[0]] # global
    ],-1)
    y_local = scatter(
        torch.cat([
            x_intersect, # id
            x_invar[data.intersect_indicator] # (local,global)->local
            ],-1)
        ,data.node_map_edge_index[1],0,dim_size=len(data.target.atoms),reduce=domain_transfer_reduce)
    y_global_domain = scatter(
            x_invar,
            data.domain_map_edge_index[1],0,dim_size=data.target.num_domains,reduce=intersect_transfer_reduce)
    if combine_reps:
        ret = torch.cat([
            y_local,
            y_global_domain[data.target.domain_indicator]
        ],-1)
    else:
        ret = y_local, y_global_domain
    if return_domain_reduction:
        return ret, x_domain
    else:
        return ret

# def transfer0_1_minimal(x: Tensor, data: TransferData1):
#     r"""
#     Performs the minimum number of reductions such that the full space of linear maps 
#     is covered if a linmaps operation is performed before and after calling this.
#     """
#     # x = x[data.domain_map_edge_index[0]]
#     # x = x[data.intersect_indicator]
#     # x = x[data.node_map_edge_index[1]]
#     x = x[data.domain_map_edge_index[0][data.intersect_indicator[data.node_map_edge_index[1]]]]
#     return x

# def transfer1_1_minimal(x: Tensor, data: TransferData1, reduce: Union[str,list[str]] = 'sum'):
#     r"""
#     Performs the minimum number of reductions such that the full space of linear maps 
#     is covered if a linmaps operation is performed before and after calling this.
#     NOTE: this only has the same representation power for commutative reductions.

#     We only consider two maps:
#         - reduce then broadcast on intersection (inv)
#         - identity map for intersection (int)
#     """
#     x = x[data.node_map_edge_index[0]]
#     inv_map = scatter(x,data.intersect_indicator,0,reduce=reduce[0])
#     cat_maps = torch.cat([x,inv_map],-1)
#     y = scatter(cat_maps,data.node_map_edge_index[1],0,reduce=reduce[1])
#     return y