from torch_geometric.utils import erdos_renyi_graph, k_hop_subgraph
from time import monotonic
import torch
from torch import Tensor
from objects2 import TransferData2, atomspack2
from ptensors2 import transfer2_2_minimal_large_ptensors, linmaps0_2

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

forw, backw = profile_random_graph_khoods(64,0.05,4,32,20)

print(forw)
print(backw)

import numpy as np

print(np.mean(forw),np.std(forw))
print(np.mean(backw),np.std(backw))