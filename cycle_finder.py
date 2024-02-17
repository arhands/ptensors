from __future__ import annotations
from typing import Generic, List, Set, Tuple, TypeVar, Union
from torch import Tensor
import numpy as np

class Node:
    neighbors: Set[Node]
    def __init__(self, neighbors : Set[Node] = set()) -> None:
        self.neighbors = neighbors

    def remove(self, replace_connectivity: bool):
        r"""
        replace_connectivity: connects all previous neigbhors to eachother.
        """
        if replace_connectivity:
            for n in self.neighbors:
                for m in self.neighbors:
                    if n != m and n not in m.neighbors:
                        m.neighbors.add(n)

        for n in self.neighbors:
            n.neighbors.remove(m)

    def copy(self) -> Node:
        return Node(self.neighbors.copy())

class Graph:
    nodes: np.ndarray[Node]
    def num_nodes(self) -> int:
        return self.nodes.shape[0]
    def __init__(self, nodes: np.ndarray[Node]) -> None:
        self.nodes = nodes

    @classmethod
    def from_edge_index(cls, edge_index: Tensor, num_nodes: int) -> Graph:
        nodes = np.array([Node() for _ in range(num_nodes)],dtype=Node)
        for e in edge_index.transpose(1,0).tolist():
            nodes[e[0]].neighbors.add(e[1])
        return cls(nodes)


    def copy(self) -> Graph:
        return Graph([n.copy() for n in self.nodes])



def reduce_graph_1(G : Graph) -> Graph:
    r"""
    Reduces the graph until there are no vertices of degree <= 1.
    """
    G = G.copy()
    keeping_mask = np.ones(G.nodes,dtype=bool)
    current_list : List[Tuple[int,Node]] = list(enumerate(G.nodes))
    new_list = []
    while len(current_list) > 0:
        for idx, n in current_list:
            deg = len(n.neighbors)
            if deg <= 1:
                if deg == 1:
                    new_list.append(next(iter((n.neighbors))))

                keeping_mask[idx] = False
                n.remove(False)

        current_list.clear()
        new_list, current_list = current_list, new_list

    return Graph(G.nodes[keeping_mask])

def reduce_graph_2(G : Graph) -> Graph:
    r"""
    Reduces the graph until there are vertices of degree 2.
    Assumes there are no vertices of degree <= 1.
    """
    G = G.copy()
    keeping_mask = np.ones(G.nodes,dtype=bool)
    nodes : List[Tuple[int,Node]] = list(enumerate(G.nodes))
    for idx, n in nodes:
        deg = len(n.neighbors)
        if deg == 2:
            m = next(iter(n.neighbors))
            deg_m = len(m.neighbors)
            if deg_m == 2:
                keeping_mask[idx] = False
                n.remove(True)

    return Graph(G.nodes[keeping_mask])


T = TypeVar('T')
class LinkedList(Generic[T]):
    r"""
    NOTE: by default, this enumerates items backwards.
    """
    last_ele: Union[LinkedList,None]
    value: T

    def __init__(self, last_ele: Union[LinkedList,None], value: T) -> None:
        super().__init__()
        self.last_ele = last_ele
        self.value = value

    def to_list(self) -> List[T]:
        if self.last_ele is None:
            return [self.value]
        else:
            l = self.last_ele.to_list()
            l.append(self.value)
            return l
    @classmethod
    def from_list(cls,ls: list[T]):
        val = None
        for v in ls:
            val = cls(val,v)
        return val

    def __getitem__(self, idx: slice) -> LinkedList[T]:
        # TODO: fix placeholder code.
        return self.from_list(self.to_list()[idx])

    def __contains__(self, key: T) -> bool:
        return key == self.value or self.last_ele is not None and key in self.last_ele

    def __str__(self) -> str:
        return str(self.to_list())


def get_smallest_enclosing_cycle(node: Node) -> LinkedList[Node]:
    # TODO: handle tie breakers.
    # uniform search
    queue = [LinkedList[Node](None,node)]
    while len(queue) > 0:
        path = queue.pop(0)
        for m in path.value.neighbors:
            if m == node:
                return path
            elif m not in path.value:
                queue.append(LinkedList[Node](path,m))
    return None

def compute_3_regular_cell_complex_3(G: Graph) -> Set[Tuple[Node]]:
    r"""
    Assumes minimal degree is 3.
    Greedly embeds all vertices into small cycles.
    """
    minimal_cycles = {tuple(get_smallest_enclosing_cycle(n).to_list()) for n in G.nodes}
    return minimal_cycles