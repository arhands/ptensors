from __future__ import annotations
import torch
from torch import Tensor
from typing import Literal, TypeAlias, Union, List, Tuple, Dict
import ptens as p
from ptens_base import atomspack
import os

ptensors_layer_type : TypeAlias = Union[p.ptensors0,p.ptensors1,p.ptensors2]
domain_list_type : TypeAlias = Union[atomspack,List[List[int]]]

def _scalar_mult(x: ptensors_layer_type, alpha: Tensor):
    return x.mult_channels(alpha.broadcast_to(x.get_nc()).clone())

def _sum(x: List[ptensors_layer_type]) -> ptensors_layer_type:
    tot = x[0]
    # nc = x[0].get_nc()
    # num_atoms = len(x[0].get_atoms())
    for i in range(1,len(x)):
        # assert len(x[i].get_atoms()) == num_atoms
        # assert nc == x[i].get_nc()
        tot = tot + x[i]
    return tot

def get_run_path(base_dir: str = 'logs') -> str:
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        run_id = 0
    else:
        contents = os.listdir(base_dir)
        contents = [int(c) for c in contents]
        run_id = 1 + max(*contents) if len(contents) > 1 else 1 + contents[0]
    run_path = f"{base_dir}/{run_id}/"
    os.mkdir(run_path)
    return run_path

def str_to_graph(name: str) -> str:
    r"""
    Takes in a string of the form "[KPC]_[1-9][0-9]*" and computes the following:
        First letter:
            K: complete graph
            P: path
            C: cycle
        Subscript: number of nodes in returned graph.
    """
    graph_type : Literal['K','P','C']= name[0]
    num_nodes = int(name[2:])
    if graph_type == 'K':
        return p.graph.from_matrix(torch.ones(num_nodes,num_nodes,dtype=torch.float32) - torch.eye(num_nodes))
    else:
        if graph_type == 'P':
            edge_index = torch.stack([torch.arange(num_nodes-1),torch.arange(1,num_nodes)])
        elif graph_type == 'C':
            edge_index = torch.stack([torch.arange(num_nodes),torch.arange(1,num_nodes + 1) % num_nodes])
        else:
            raise Exception(f'Graph type "{graph_type}" not recognized.')
        edge_index = edge_index.float()
        edge_index = torch.cat([edge_index,edge_index.flip(0)],-1)
        return p.graph.from_edge_index(edge_index)

class GraphMapCache:
    r"""
    Base graph is always called 'G'.
    """
    
    store: Dict[Tuple[str,str],p.graph]
    
    atoms: Dict[str,atomspack]
    r"""Entry is None for when there are no atoms."""

    base_graph: p.graph

    def __init__(self, base_graph: p.graph, patterns: Dict[str,p.graph], atoms: Dict[str,p.graph]) -> None:
        self.base_graph = base_graph
        self.store = {('G','G') : base_graph}
        self.atoms = {
            **{k : base_graph.subgraphs(patterns[k]) for k in patterns},
            **atoms,
            'G' : base_graph.nhoods(0)
        }
        self.atoms = {k : self.atoms[k] for k in self.atoms if len(self.atoms[k]) > 0}

    def __getitem__(self, key: Union[Tuple[str,str],str]) -> Union[p.graph,None]:
        r"""
        Returns 'None' if the graph is empty.
        """
        # syntactic sugar
        if isinstance(key,str):
            return self[key,key]
        
        # checking if map has already been computed
        if key in self.store:
            return self.store[key]
        
        # checking if map is non-empty.
        if key[0] in self.atoms and key[1] in self.atoms:
            # computing map
            graph_map = p.graph.overlaps(self.atoms[key[0]],self.atoms[key[1]])
            self.store[key] = graph_map
            return graph_map
        else:
            return None