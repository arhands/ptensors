from torch_geometric.utils import erdos_renyi_graph, k_hop_subgraph
from time import monotonic
import torch
from torch import Tensor
from objects2 import TransferData2
from ptensors2 import transfer2_2_minimal_large_ptensors
from torch_geometric.data import Data

def prepare_graph(edge_index: Tensor) -> TransferData2:
    

def profile_random_graph_khoods(num_nodes: int, edge_prob: int, num_rounds: int, checks_per_round) -> Tensor:

