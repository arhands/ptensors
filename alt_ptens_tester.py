from torch_geometric.datasets import ZINC
from induced_cycle_finder import get_induced_cycles, from_edge_index
from tqdm import tqdm

# print(graph.edge_index)

# from cycle_finder import Node, Graph


# import numpy as np
# neighbors_list = [
#     [5,1],
#     [0,2],
#     [1,3],
#     [2,4],
#     [3,5],
#     [4,0]
# ]
# neighbors = np.empty(6,dtype=object)
# for i in range(len(neighbors_list)):
#     neighbors[i] = neighbors_list[i]


# neighbors_list = [
#     [3,1,2], # 0
#     [0,2],   # 1
#     [1,3,0], # 2
#     [2,0],   # 3
# ]
# neighbors = np.empty(len(neighbors_list),dtype=object)
# for i in range(len(neighbors_list)):
#     neighbors[i] = neighbors_list[i]

import numpy as np
counts = []

ds = ZINC('./data/ZINC',True,'val')
G = ds[0]
neighbors = from_edge_index(G.edge_index,G.num_nodes)

neighbors_list = [
    [3,1,2], # 0
    [0,2],   # 1
    [1,3,0], # 2
    [2,0,4],   # 3
    [3],   # 4
]
neighbors = np.empty(len(neighbors_list),dtype=object)
for i in range(len(neighbors_list)):
    neighbors[i] = neighbors_list[i]

induced_cycles = get_induced_cycles(neighbors)
induced_cycles = [v.to_list() for v in induced_cycles]

from objects import atomspack, TransferData1
from ptensors1 import linmaps0_1, linmaps1_0, linmaps1_1, transfer0_1, transfer1_0, transfer1_1
import torch

cycle_pack = atomspack.from_list(induced_cycles)

edges = []
for idx, N in enumerate(neighbors_list):
    edges.extend([[idx,j] for j in N])
edge_pack = atomspack.from_list(edges)

transfer_data = TransferData1.from_atomspacks(cycle_pack,edge_pack)

cycles = torch.tensor([[10**i,2*10**i] for i in range(len(induced_cycles))])
print(cycle_pack)
print(edge_pack)
print(transfer0_1(cycles,transfer_data))

edges = torch.tensor([[10**i,2*10**i] for i in range(len(edges))])
edges = torch.tensor([[10**i] for i in range(len(edges))])
# print(edges)
# print()
transfer0_1(edges,TransferData1.from_atomspacks(edge_pack,cycle_pack))
# print(cycle_pack)
linmaps0_1(edges,edge_pack)

# ptensors1
transfer_map = TransferData1.from_atomspacks(cycle_pack,edge_pack)

cycles = torch.tensor([[10**n] for n in range(len(cycle_pack.atoms))])
print(transfer1_1(cycles,transfer_map))