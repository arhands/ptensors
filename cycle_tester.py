import math
from torch_geometric.datasets import LRGBDataset
from graph_property import GraphPropertyDataset
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


counts = []

outer_loop = tqdm(['train','val','test'],total=3)

for split in outer_loop:
    outer_loop.set_description_str(split)
    ds = LRGBDataset('.','peptides-struct',split)
    # ds = ZINC('./data/ZINC_base',True,split)
    loop = tqdm(ds,total=len(ds),leave=False,position=1)
    for G in loop:
        neighbors = from_edge_index(G.edge_index,G.num_nodes)
        induced_cycles = get_induced_cycles(neighbors,math.inf)
        lengths = [len(c.to_list()) for c in induced_cycles]
        if len(lengths) > 0:
            max_length = max(lengths)
            while len(counts) <= max_length:
                counts.append(0)
            for l in lengths:
                counts[l] += 1

print("lengths:")
for idx, c in enumerate(counts):
    if c > 0:
        print("\t%3d : %d" % (idx,c))