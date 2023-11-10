from __future__ import annotations
import torch
from torch import Tensor
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.data import Data
from objects import atomspack2_minimal, atomspack3_minimal

# def linmaps1_2_strict(x: Tensor, domains: atomspack2) -> list[Tensor]:
#     r"""Excludes invariant maps (i.e., reductions that reduce to ptensors0 layers)."""


def internal_scaled_dot_product_attention_1(Q: Tensor, K: Tensor, V: Tensor, data: atomspack2_minimal):
    # TODO: for correctness, check that the indicators are consistent with the original paper.
    # They should be correct enough, but it may be that Q and K need to be swapped.
    products = torch.bmm(Q[data.row_indicator],K[data.col_indicator]) / torch.sqrt(Q.size(1))
    probabilities = scatter_softmax(products,data.row_indicator,0)
    results = scatter_sum(V[data.row_indicator]*probabilities,data.row_indicator,0,dim_size=V.size(0))
    return results

def internal_scaled_dot_product_attention_2(Q: Tensor, K: Tensor, V: Tensor, data: atomspack3_minimal):
    # TODO: for correctness, check that the indicators are consistent with the original paper.
    # They should be correct enough, but it may be that Q and K need to be swapped.
    products = torch.bmm(Q[data.ij_to_ijk_indicator],K[data.ij_to_ikj_indicator]) / torch.sqrt(Q.size(1))
    probabilities = scatter_softmax(products,data.ij_to_ijk_indicator,0)
    results = scatter_sum(V[data.ij_to_ijk_indicator]*probabilities,data.ij_to_ijk_indicator,0,dim_size=V.size(0))
    return results