import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, BatchNorm1d
from objects import TransferData1
from torch_scatter import scatter_sum

# class LinmapsTransfer1_1(Module):
#     def __init__(self, hidden_units: int, y_to_x: bool, x_to_x: bool) -> None:
#         super().__init__()
#         assert not y_to_x or x_to_x

#         self.y2x = y_to_x
#         self.x2x = x_to_x
#         self.hidden_units = hidden_units
#         omni_slf_dim = 2*hidden_units if y_to_x or x_to_x else hidden_units
#         omni_dim = 2*hidden_units if y_to_x else hidden_units
#         self.omni_dim = omni_dim
#         self.inv_dim = omni_dim*2

#         self.x_sum_lin = Linear(hidden_units,omni_slf_dim + omni_dim,False)
#         self.x_lin = Linear(hidden_units,omni_slf_dim + omni_dim + hidden_units,False)
#         self.y_sum_lin = Linear(hidden_units,omni_dim,False)
#         self.y_lin = Linear(hidden_units,omni_dim,False)
#         self.msg_bn = BatchNorm1d(omni_dim)

#         self.inv_msg_bn = BatchNorm1d(omni_dim)

#     r"""
#     Performs both linmaps and transfer operations on both the source and target with "edge updates".
#     """
#     def forward(self, x: Tensor, y: Tensor, x_to_y: TransferData1):
#         # msg (intersect)

#         x_sum = scatter_sum(x,x_to_y.source.domain_indicator,0)
#         x_sum = self.x_sum_lin(x_sum)
#         x = self.x_lin(x)
#         x_msg, x_sum = x_sum[x_to_y.source.domain_indicator,self.omni_dim:] + x
#         y_sum = scatter_sum(y,x_to_y.target.domain_indicator,0)
#         y_sum = self.y_lin(y_sum)
#         y = self.y_lin(y)
#         y_msg = y_sum[x_to_y.target.domain_indicator] + y

#         msg = x_msg[x_to_y.node_map_edge_index[0]] + y_msg[x_to_y.node_map_edge_index[1]]
#         msg, int_inv = msg[:,:-self.inv_dim], msg[:,-self.inv_dim:]
#         int_inv = scatter_sum(int_inv,x_to_y.intersect_indicator,0)
#         if self.x2x:
#             msg, x2x = msg[:,self.hidden_units:], msg[:,:self.hidden_units]
#         msg, int_inv = msg + int_inv[x_to_y.intersect_indicator,self.omni_dim:], int_inv[:,:self.omni_dim]

#         # msg (non-intersect)
#         x2y_inv_msg = self.inv_msg_bn(
#             self.inv_msg_x_lin(x_sum)[x_to_y.domain_map_edge_index[0]] + 
#             self.inv_msg_x_lin(y_sum)[x_to_y.domain_map_edge_index[1]])
        
#         # y2x
#         if self.y2x:

class Transfer1_0(Module):
    def __init__(self, hidden_units: int) -> None:
        super().__init__()
        self.hidden_units = hidden_units

        self.x_sum_lin = Linear(hidden_units,hidden_units,False)
        self.x_int_lin = Linear(hidden_units,hidden_units,False)
        self.y_lin = Linear(hidden_units,hidden_units*2,False)
        self.msg_bn = BatchNorm1d(hidden_units)

    r"""
    Performs both linmaps and transfer operations on both the source and target with "edge updates".
    """
    def forward(self, x: Tensor, y: Tensor, x_to_y: TransferData1):
        # msg

        x_sum = scatter_sum(x,x_to_y.source.domain_indicator,0)
        x_sum = self.x_sum_lin(x_sum)
        x_int = scatter_sum(x[x_to_y.node_map_edge_index[0]],x_to_y.intersect_indicator,0)
        x_int = self.x_int_lin(x_int)
        x_msg = x_sum[x_to_y.domain_map_edge_index[0]] + x_int
        
        y_msg = y[x_to_y.domain_map_edge_index[1]]
        y_msg = self.y_lin(y_msg)
        
        msg = x_msg + y_msg
        msg = self.msg_bn(msg).relu()

        res = scatter_sum(msg,x_to_y.domain_map_edge_index[1],0)

        return res

class Transfer0_1(Module):
    def __init__(self, hidden_units: int) -> None:
        super().__init__()
        self.hidden_units = hidden_units

        self.x_sum_lin = Linear(hidden_units,hidden_units,False)
        self.x_int_lin = Linear(hidden_units,hidden_units,False)
        self.x_lin = Linear(hidden_units,hidden_units*2,False)
        self.msg_bn = BatchNorm1d(hidden_units)

    r"""
    Performs both linmaps and transfer operations on both the source and target with "edge updates".
    """
    def forward(self, x: Tensor, y: Tensor, x_to_y: TransferData1):
        # msg

        y_sum = scatter_sum(x,x_to_y.source.domain_indicator,0)
        y_sum = self.x_sum_lin(y_sum)
        y_int = scatter_sum(x[x_to_y.node_map_edge_index[1]],x_to_y.intersect_indicator,0)
        y_int = self.x_int_lin(y_int)
        y_msg = y_sum[x_to_y.domain_map_edge_index[1]] + y_int
        
        x_msg = x[x_to_y.domain_map_edge_index[0]]
        x_msg = self.x_lin(x_msg)
        
        msg = y_msg + x_msg
        msg = self.msg_bn(msg).relu()

        return msg
    