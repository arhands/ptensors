# TODO: replace with https://arxiv.org/pdf/1509.06464.pdf
from __future__ import annotations
from typing import Any, Callable, Union
from cycle_finder import Node, LinkedList
import numpy as np

# from torch import Tensor
# from torch_scatter import scatter_min

class SpanningTree:
    parents: np.ndarray
    children: list[list[int]]
    unused_edge_graph: list[list[int]]
    def edge_in_tree(self, i: int, j: int):
        return self.parents[i] == j or self.parents[j] == i
    
    def DFS_down(self, root_idx: int, fn: Callable[[int],bool]) -> bool:
        r"""performs depth-first search in the component specified by 'root_idx' until 'fn' returns True."""
        if fn(root_idx):
            return True
        else:
            for c in self.children[root_idx]:
                if self.DFS_down(c,fn):
                    return True
        return False
    
    def remove_edge(self, i: int, j: int, in_place: bool) -> SpanningTree:
        r"""NOTE: only copies elements in a way that leaves the spanning tree being acted on unmodified, but future in_place operations may still augment this object."""
        if self.parents[j] == i:
            i,j = j,i
        elif self.parents[i] != j:
            if in_place:
                val = self
            else:
                val = SpanningTree()
                val.parents = self.parents
                val.children = self.children
                unused_edge_graph = self.unused_edge_graph.copy()
                unused_edge_graph[i] = unused_edge_graph[i].copy()
                unused_edge_graph[j] = unused_edge_graph[j].copy()
                val.unused_edge_graph = unused_edge_graph
            val.unused_edge_graph[i].remove(j)
            val.unused_edge_graph[j].remove(i)
            return self
        
        component_id = np.zeros(len(self.children),dtype=bool)
        descendent_nodes = []
        def mark(idx: int):
            component_id[idx] = True
            for c in self.children[idx]:
                mark(c)
        mark(i)
        
        def get_connection(idx: int):
            self.
        



def _compute_minimum_spanning_tree_prim(root_idx: int, nodes: list[np.ndarray]) -> list[int]:
    r"""Prim's algorithm for an unweighted graph rooted at node 'root_idx'."""
    unvisited = np.ones(len(nodes),dtype=bool)
    unvisited[root_idx] = False
    
    queue = [LinkedList[int](None,root_idx)]

    # Paths represents a bottom-up spanning tree.
    parents = np.empty(len(nodes),dtype=np.int32)
    parents[root_idx] = -1
    
    while len(queue) > 0:
        path = queue.pop(0)
        unvisited_neighbors = nodes[path.value]
        unvisited_neighbors = unvisited_neighbors[unvisited[unvisited_neighbors]]
        unvisited[unvisited_neighbors] = False
        new_paths = [LinkedList[int](path,v) for v in unvisited_neighbors]
        queue.extend(new_paths)
        parents[unvisited_neighbors] = path.value
    return parents


def _remove_edge(spanning_tree: SpanningTree, i: int, j: int) -> tuple[SpanningTree,bool]:
    if i not in spanning_tree.parents[j]:
        return spanning_tree, False
    
    
