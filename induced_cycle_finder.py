# based on https://arxiv.org/pdf/1408.1265.pdf
from typing import Any, Optional, Union

from torch import Tensor
from cycle_finder import LinkedList
import numpy as np 

def _remove_node(idx: int, neighbors: np.ndarray[Any,list[int]]):
    for n in neighbors[idx]:
        neighbors[n].remove(idx)
    
    neighbors[idx] = []

def from_edge_index(edge_index: Tensor, num_nodes: int) -> np.ndarray[Any,list[int]]:
    neighbors = np.array([[] for _ in range(num_nodes)],dtype=object)
    neighbors = np.empty(num_nodes,dtype=object)
    for i in range(num_nodes):
        neighbors[i] = []
    for e in edge_index.transpose(1,0).tolist():
        neighbors[e[0]].append(e[1])
    return neighbors

# NOTE: we should replace this with a spanning tree.
def _get_shortest_path(a: int, b: int, neighbors: np.ndarray[Any,list[int]], active: Optional[np.ndarray[Any,list[int]]] = None) -> Union[LinkedList[int],None]:
    unvisited = active.copy() if active is not None else np.ones(neighbors.shape[0],dtype=bool)
    queue = [LinkedList[int](None,a)]
    unvisited[a] = False
    while len(queue) > 0:
        path = queue.pop(0)
        if path.value == b:
            return path
        unvisited_neighbors = [v for v in neighbors[path.value] if unvisited[v]]
        unvisited[unvisited_neighbors] = False
        queue.extend([LinkedList[int](path,v) for v in unvisited_neighbors])
    return None

def list_induced_paths(s: int, t: int, pi_su: LinkedList[int], u: int, pi_ut: LinkedList[int], neighbors: np.ndarray[Any,list[int]], active_nodes: np.ndarray[Any,bool]) -> list[LinkedList[int]]:
    assert all(active_nodes[v] for v in pi_ut.to_list())
    # active_nodes = np.ones(neighbors.shape[0],dtype=bool)
    if u == t:
        return [pi_su]
    else:
        S : list[tuple[int,LinkedList[int]]]= []
        while True:
            # Getting closest vertex to t in the path from u to t that is in the neighborhood of u.
            path = pi_ut
            assert path is not None
            dist = 0
            while path is not None:
                if path.value in neighbors[u]:
                    v = path.value
                    pi_vt = pi_ut[-dist-1:]
                    break
                path = path.last_ele
                dist += 1
            # assert active_nodes[v]
            active_nodes[v] = False
            S.append((v,pi_vt))

            pi_ut = _get_shortest_path(u,t,neighbors,active_nodes)
            if pi_ut is None:
                break
        paths = []
        for v, pi_vt in S:
            active_nodes[v] = True
            paths.extend(list_induced_paths(s, t, LinkedList[int](pi_su,v),v,pi_vt,neighbors,active_nodes))

            active_nodes[v] = False
        # performing housekeeping by adding back all of the removed nodes.
        active_nodes[[v for v, _ in S]] = True
        return paths

def get_induced_cycles(neighbors: np.ndarray[Any,list[int]]) -> list[LinkedList[int]]:
    r"""
    NOTE: we assume the graph is undirected.
    """
    cycles : list[LinkedList[int]] = []
    num_nodes = neighbors.shape[0]
    active_nodes = np.ones(num_nodes,bool)
    for s in range(num_nodes):
        for t in neighbors[s].copy():
            neighbors[s].remove(t)
            neighbors[t].remove(s)
            initial_path = _get_shortest_path(s,t,neighbors,active_nodes)
            if initial_path is not None:
                cycles.extend(list_induced_paths(s,t,LinkedList[int](None,s),s,initial_path,neighbors,active_nodes))
            neighbors[s].append(t)
            neighbors[t].append(s)
            active_nodes[t] = False
        active_nodes[neighbors[s]] = True
        _remove_node(s,neighbors)
    return cycles

# 98 [list([3, 1]) list([0, 2]) list([1, 3]) list([2, 0])] 0 3
# 98 [list([]) list([0]) list([1]) list([2])] 1 0