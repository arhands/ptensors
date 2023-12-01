from torch_geometric.utils import erdos_renyi_graph, k_hop_subgraph
from time import monotonic
import torch
from torch import Tensor
from objects2 import TransferData2, atomspack2
from ptensors2 import transfer2_2_minimal_large_ptensors, linmaps0_2
import pandas as pd
import ptens
from ptens_base import atomspack as ptens_atomspack
torch.manual_seed(5)

def profile_random_graph_khoods(num_nodes: int, edge_prob: float, num_hops: int, num_channels, num_rounds: int) -> Tensor:
    edge_index = erdos_renyi_graph(num_nodes,edge_prob)
    # getting subgraphs
    # return
    subg = [
        k_hop_subgraph(idx,num_hops,edge_index,num_nodes=num_nodes)[0]
        for idx in range(num_nodes)
    ]
    print(min(len(k) for k in subg))
    features = torch.rand(num_nodes,num_channels,device='cuda')

    # preprocessing
    subgraphs = atomspack2.from_tensor_list(subg)
    transfer = TransferData2.from_atomspacks(subgraphs,subgraphs)
    transfer.to('cuda')
    features2 = linmaps0_2(features,subgraphs)
    features2.requires_grad = True
    print("preprocessing complete.")
    # test = torch.compile(transfer2_2_minimal_large_ptensors)
    # transfer = TransferData2.from_atomspacks(subgraphs,subgraphs,False)
    

    # running tests
    forward_history = []
    backward_history = []
    
    for _ in range(num_rounds):
        features2_ = features2.clone()
        t0 = monotonic()
        # y = test(features2_,transfer)
        y = transfer2_2_minimal_large_ptensors(features2_,transfer)
        t1 = monotonic()
        y.sum().sigmoid().backward()
        t2 = monotonic()
        torch.cuda.empty_cache()
        forward_history.append(t1 - t0)
        backward_history.append(t2 - t1)
    return forward_history, backward_history

def profile_random_domains(num_nodes: int, ptensor_size: int, num_ptensors: int, num_channels, num_rounds: int, use_ptens: bool) -> Tensor:
    # getting subgraphs
    # return
    subg = [
        torch.randperm(num_nodes)[:ptensor_size]
        for _ in range(num_ptensors)
    ]
    print(min(len(k) for k in subg))
    features = torch.rand(num_nodes,num_channels,device='cuda')

    if not use_ptens:
        # preprocessing
        subgraphs = atomspack2.from_tensor_list(subg)
        transfer = TransferData2.from_atomspacks(subgraphs,subgraphs)
        transfer.to('cuda')
        features2 = linmaps0_2(features,subgraphs)
        features2.requires_grad = True
    else:
        atoms = [
            v.tolist() for v in subg
        ]
        atoms = ptens_atomspack(atoms)
        features.requires_grad = True
    print("preprocessing complete.")
    # test = torch.compile(transfer2_2_minimal_large_ptensors)
    # transfer = TransferData2.from_atomspacks(subgraphs,subgraphs,False)
    

    # running tests
    forward_history = []
    backward_history = []
    features_ = features
    if use_ptens:
        features = features_.clone()
        features = ptens.ptensors0.from_matrix(features,atoms)
        features = ptens.ptensors0.linmaps2(features)
    for _ in range(num_rounds):
        if use_ptens:
            # features = torch.rand(num_nodes,num_channels,requires_grad=True,device='cuda')
            print("line 86",flush=True)
            print("line 88",flush=True)
            print("line 90",flush=True)
        else:
            features2_ = features2.clone()
        t0 = monotonic()
        # y = test(features2_,transfer)
        if use_ptens:
            overlaps = ptens.graph.overlaps(atoms,atoms)
            # print("line 94",flush=True)
            y = ptens.ptensors2.transfer2(features,atoms,overlaps)
            # print("line 96",flush=True)
        else:
            y = transfer2_2_minimal_large_ptensors(features2_,transfer)
        t1 = monotonic()
        if use_ptens:
            y = y.torch()
        # print("line 102",flush=True)
        y.sigmoid().sum().sigmoid().backward()
        # y.sum().sigmoid().backward()
        # print("line 104",flush=True)
        t2 = monotonic()
        # torch.cuda.empty_cache()
        forward_history.append(t1 - t0)
        backward_history.append(t2 - t1)
        print(t1 - t0, t2 - t1)

    return forward_history, backward_history

# forw, backw = profile_random_graph_khoods(2,0.05,4,32,20)
forw, backw = profile_random_domains(256,128,32,32,10,False)
# forw, backw = profile_random_domains(256,128,32,32,20,False)

print(forw)
print(backw)

import numpy as np

print(np.mean(forw),np.std(forw))
print(np.mean(backw),np.std(backw))

""" python
[0.24760839994996786, 1.2782250000163913, 1.2584234001114964, 1.3258094999473542, 1.2472391000483185, 1.3065166999585927, 1.2589549999684095, 1.307321600150317, 1.2533862998243421, 1.3126604999415576]
[0.038752200081944466, 0.001758800121024251, 0.0017167998012155294, 0.0015958999283611774, 0.003233599942177534, 0.001692499965429306, 0.0035621998831629753, 0.001769399968907237, 0.0016344999894499779, 0.001731000142171979]
1.1796145499916748 0.31182815969539324
0.005744689982384444 0.011023224473550388
"""

""" ptens
[0.39550450001843274, 2.317338600056246, 2.250502600101754, 2.315676199970767, 2.2803662000223994, 2.2671245001256466, 2.2966615001205355, 2.2591425999999046, 2.3224092999007553, 2.331955200061202]
[4.191709300037473, 2.6015183001291007, 2.520332899875939, 2.4480571998283267, 2.455493099987507, 2.445179499918595, 2.5153024999890476, 2.599102099891752, 2.8299000000115484, 2.5275918999686837]
2.1036681200377645 0.570026970440159
2.713418679963797 0.5044263414875151"""